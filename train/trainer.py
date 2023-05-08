from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess

import gradio as gr
import torch

from .utils import (
    load_pretrained,
    prepare_args_from_dict,
    prepare_data,
    preprocess_data,
    plot_loss,
    Seq2SeqDataCollatorForChatGLM,
    ComputeMetrics,
    Seq2SeqTrainerForChatGLM
)


class GLMTrainer:
    def __init__(self):
        self.is_running = False
        self.is_running_message = "Another training is in progress."

        self.output_dir = pathlib.Path("results")
        self.instance_data_dir = self.output_dir / "training_data"

        print(self.instance_data_dir)

    def check_if_running(self) -> dict:
        if self.is_running:
            return gr.update(value=self.is_running_message)
        else:
            return gr.update(value="No training is running.")

    def cleanup_dirs(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def prepare_dataset(self, dataset_files: list) -> None:
        self.instance_data_dir.mkdir(parents=True)

        dataset_names = []
        for i, temp_path in enumerate(dataset_files):
            filename = os.path.split(temp_path.name)[-1]
            shutil.move(temp_path.name, self.instance_data_dir / filename)
            dataset_names.append(filename)
        return dataset_names
        


    def run(
        self,
        base_model: str,
        n_steps: int,
        dataset_files: list | None,
        learning_rate: float,
        gradient_accumulation: int,
        fp16: bool,
        use_8bit_adam: bool,
        gradient_checkpointing: bool,
        with_prior_preservation: bool,
        prior_loss_weight: float,
        lora_r: int,
        lora_alpha: int,
        lora_bias: str,
        lora_dropout: float,
        lora_text_encoder_r: int,
        lora_text_encoder_alpha: int,
        lora_text_encoder_bias: str,
        lora_text_encoder_dropout: float,
    ) -> tuple[dict, list[pathlib.Path]]:
        if not torch.cuda.is_available():
            raise gr.Error("CUDA is not available.")

        if self.is_running:
            return gr.update(value=self.is_running_message), []

        if dataset_files is None:
            raise gr.Error("You need to upload dataset files.")

        
        self.cleanup_dirs()
        dataset_names = self.prepare_dataset(dataset_files)
        dataset_str = ",".join(dataset_names)

        param_dict = {
            "do_train": 1,
            "dataset": dataset_str,
            "dataset_dir": self.instance_data_dir,
            "finetuning_type": "lora",
            "output_dir": self.output_dir,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 1000,
            "learning_rate": 5e-5,
            "num_train_epochs": 1.0,
            "fp16": 1
        }
        model_args, data_args, training_args, finetuning_args = prepare_args_from_dict(param_dict)
        dataset = prepare_data(model_args, data_args)
        model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="sft")
        dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft")
        data_collator = Seq2SeqDataCollatorForChatGLM(
            tokenizer=tokenizer,
            model=model,
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
            inference_mode=(not training_args.do_train)
        )

        # Override the decoding parameters of Trainer
        training_args.generation_max_length = training_args.generation_max_length if \
                    training_args.generation_max_length is not None else data_args.max_target_length
        training_args.generation_num_beams = data_args.num_beams if \
                    data_args.num_beams is not None else training_args.generation_num_beams

        # Initialize our Trainer
        trainer = Seq2SeqTrainerForChatGLM(
            finetuning_args=finetuning_args,
            model=model,
            args=training_args,
            train_dataset=dataset if training_args.do_train else None,
            eval_dataset=dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None
        )

        # Keyword arguments for `model.generate`
        gen_kwargs = {
            "do_sample": True,
            "top_p": 0.7,
            "max_length": 768,
            "temperature": 0.95
        }

        self.is_running = True
        # Training
        if training_args.do_train:
            train_result = trainer.train()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state() # along with the loss values
            trainer.save_model()
            if finetuning_args.plot_loss:
                plot_loss(training_args)

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if training_args.do_predict:
            predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(predict_results, tokenizer)

        self.is_running = False

        # todo 判断训练结果
        result_message = "Training Completed!"

        weight_paths = sorted(self.output_dir.glob("*.pt"))
        config_paths = sorted(self.output_dir.glob("*.json"))
        return gr.update(value=result_message), weight_paths + config_paths