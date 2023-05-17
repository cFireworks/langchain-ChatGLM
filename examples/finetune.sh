#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./finetune.py \
    --do_train \
    --dataset auto_generated_api_data.json \
    --dataset_dir ./results/training_data \
    --finetuning_type lora \
    --output_dir ./results/openplatform-api-4w-515 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 3e-4 \
    --num_train_epochs 3.0 \
    --fp16
