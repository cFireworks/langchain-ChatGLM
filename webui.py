import gradio as gr
import os
import shutil
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
from typing import List
import pathlib
import PIL.Image
import shortuuid
from train.utils import other

from train.inference import InferencePipeline
from train.trainer import GLMTrainer

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3

LATEST_CHECKPOINT = "openplatform-api-4w-515"

def get_vs_list():
    if not os.path.exists(VS_ROOT_PATH):
        return []
    return os.listdir(VS_ROOT_PATH)


def get_fine_tuned_model_list():
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        return []
    sub_list = os.listdir(FINE_TUNED_MODEL_PATH)
    sub_list = [f for f in sub_list if os.path.isdir(os.path.join(FINE_TUNED_MODEL_PATH, f)) and f != "training_data"]
    return sub_list


vs_list = ["新建知识库"] + get_vs_list()

fine_tuned_model_list = get_fine_tuned_model_list()

embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()






def get_answer(query, vs_path, history, mode):
    if vs_path and mode == "知识库问答":
        resp, history = local_doc_qa.get_knowledge_based_answer(
            query=query, vs_path=vs_path, chat_history=history)
        source = "".join([f"""<details> <summary>出处 {i + 1}</summary>
{doc.page_content}

<b>所属文件：</b>{doc.metadata["source"]}
</details>""" for i, doc in enumerate(resp["source_documents"])])
        history[-1][-1] += source
    else:
        resp = local_doc_qa.llm._call(query)
        history = history + [[query, resp + ("\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")]]
    return history, ""


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


def init_model():
    try:
        full_checkpoint_dir = os.path.join(FINE_TUNED_MODEL_PATH, LATEST_CHECKPOINT)
        local_doc_qa.init_cfg(llm_model=LLM_MODEL,
                              embedding_model=EMBEDDING_MODEL,
                              llm_history_len=3,
                              checkpoint_dir=full_checkpoint_dir,
                              use_ptuning_v2=False,
                              use_lora=False,
                              top_k=6)
        local_doc_qa.llm._call("你好")
        return """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        return """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""


def reinit_model(llm_model, embedding_model, llm_history_len, use_ptuning_v2, use_lora, top_k, history, checkpoint_dir):
    try:
        full_checkpoint_dir = os.path.join(FINE_TUNED_MODEL_PATH, checkpoint_dir)
        local_doc_qa.init_cfg(llm_model=llm_model,
                              embedding_model=embedding_model,
                              llm_history_len=llm_history_len,
                              checkpoint_dir=full_checkpoint_dir,
                              use_ptuning_v2=use_ptuning_v2,
                              use_lora=use_lora,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]


def remove_model():
    local_doc_qa.clear_model()
    torch.cuda.empty_cache()


def get_vector_store(vs_id, files, history):
    vs_path = VS_ROOT_PATH + vs_id
    filelist = []
    for file in files:
        filename = os.path.split(file.name)[-1]
        shutil.move(file.name, UPLOAD_ROOT_PATH + filename)
        filelist.append(UPLOAD_ROOT_PATH + filename)
    if local_doc_qa.llm and local_doc_qa.embeddings:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path)
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), VS_ROOT_PATH + vs_id


def change_mode(mode):
    if mode == "知识库问答":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def change_train_mode(mode):
    if mode == "lora":
        pass
    else:
        pass


def add_fm_name(fm_name, fm_list):
    if fm_name in fm_list:
        fm_name = fm_name+'_'+shortuuid.uuid()

    return gr.update(visible=True, choices= [fm_name] + fm_list, value=fm_name), [fm_name] + fm_list


def add_vs_name(vs_name, vs_list, chatbot):
    if vs_name in vs_list:
        chatbot = chatbot + [[None, "与已有知识库名称冲突，请重新选择其他名称后提交"]]
        return gr.update(visible=True), vs_list, chatbot
    else:
        chatbot = chatbot + [
            [None, f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """]]
        return gr.update(visible=True, choices=vs_list + [vs_name], value=vs_name), vs_list + [vs_name], chatbot


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}

.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉Private-ChatGLM WebUI🎉

"""

init_message = """欢迎使用 Private-ChatGLM Web UI！

请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

知识库问答模式中，选择知识库名称后，即可开始问答，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。

知识库暂不支持文件删除，该功能将在后续版本中推出。
"""

def update_train_loss_image(fintume_model_name) -> PIL.Image:
    print("update_train_loss_image:" + "/mnt/workspace/langchain-ChatGLM/results/" + fintume_model_name)
    return other.plot_loss_to_image("/mnt/workspace/langchain-ChatGLM/results/" + fintume_model_name)


def update_output_files(fintume_model_name) -> dict:
    paths = sorted(pathlib.Path("results/fintume_model_name").glob("*.bin"))
    config_paths = sorted(pathlib.Path("results/fintume_model_name").glob("*.json"))
    paths = paths + config_paths
    paths = [path.as_posix() for path in paths]  # type: ignore
    return gr.update(value=paths or None)


def find_weight_files() -> List[str]:
    curr_dir = pathlib.Path(__file__).parent
    paths = sorted(curr_dir.rglob("*.bin"))
    return [path.relative_to(curr_dir).as_posix() for path in paths]


def reload_lora_weight_list() -> dict:
    return gr.update(choices=find_weight_files())


def create_training_demo(trainer: GLMTrainer, pipe: InferencePipeline) -> gr.Blocks:
    with gr.Blocks() as demo:
        base_model = gr.Dropdown(
            choices=[
                "THUDM/chatglm-6b",
            ],
            value="THUDM/chatglm-6b",
            label="基础模型",
            visible=True,
        )
        resolution = gr.Dropdown(choices=["512"], value="512", label="Resolution", visible=False)

        with gr.Row():
            with gr.Box():
                gr.Markdown("训练数据")
                dataset_files = gr.Files(label="上传训练数据")
                gr.Code(
                    """
                    // 上传训练数据文件为.json格式，示例如下.
                    [
                        {
                            "instruction": "听起来很不错。人工智能可能在哪些方面面临挑战呢？",
                            "input": "",
                            "output": "人工智能面临的挑战包括数据隐私、安全和道德方面的问题，以及影响就业机会的自动化等问题。",
                            "history": [
                                [
                                    "你好，你能帮我解答一个问题吗？", 
                                    "当然，请问有什么问题？"],
                                [
                                    "我想了解人工智能的未来发展方向，你有什么想法吗？", 
                                    "人工智能在未来的发展方向可能包括更强大的机器学习算法，更先进的自然语言处理技术，以及更加智能的机器人。"]
                            ]
                        },
                        {
                            "instruction": "好的，谢谢你！",
                            "input": "",
                            "output": "不客气，有其他需要帮忙的地方可以继续问我。",
                            "history": [
                                [
                                    "你好，能告诉我今天天气怎么样吗？", 
                                    "当然可以，请问您所在的城市是哪里？"],
                                [
                                    "我在纽约。", 
                                    "纽约今天晴间多云，气温最高约26摄氏度，最低约18摄氏度，记得注意保暖喔。"]
                            ]
                        }
                    ]
                    """,
                    language='javascript',
                )
            with gr.Box():
                with gr.Column():
                    train_mode = gr.Dropdown(
                        choices=["lora", "p_tuning", "full", "freeze"],
                        label="请选择微调训练模式",
                        value="lora",
                        visible=True,
                    )
                    custom_model_name = gr.Textbox(value=shortuuid.uuid(), label="训练模型保存名称：", lines=1, interactive=True)
                    gr.Markdown("训练参数")
                    learning_rate = gr.Number(label="学习率", value=0.0005)
                    num_train_epochs = gr.Number(label="训练轮次", value=10, precision=0)
                    num_save_steps = gr.Number(label="保存步数间隔", value=1000, precision=0)
                    per_device_train_batch_size = gr.Number(label="Batch Size Per GPU", value=2, precision=0)
                    

                    # lora_bias = gr.Dropdown(
                    #     choices=["none", "all", "lora_only"],
                    #     value="none",
                    #     label="LoRA Bias for unet. This enables bias params to be trainable based on the bias type",
                    #     visible=True,
                    # )
                    # gradient_accumulation = gr.Number(label="Number of Gradient Accumulation", value=1, precision=0)
                    fp16 = gr.Checkbox(label="FP16", value=True)

        run_button = gr.Button("开始训练")
        with gr.Box():
            check_status_button = gr.Button("训练状态")
            with gr.Row():
                train_loss_image = gr.Image(type='pil')
                with gr.Column():
                    with gr.Box():
                        gr.Markdown("Message")
                        training_status = gr.Markdown()
                    output_files = gr.Files(label="训练权重和配置文件")

        run_button.click(fn=remove_model)
        run_button.click(fn=pipe.clear)
        run_button.click(
            fn=trainer.run,
            inputs=[
                base_model,
                dataset_files,
                learning_rate,
                num_train_epochs,
                num_save_steps,
                per_device_train_batch_size,
                fp16,
                train_mode,
                custom_model_name,
            ],
            outputs=[
                training_status,
                output_files,
            ],
            queue=False,
        )
        if output_files.file_types is not None:
            run_button.click(
                fn=add_fm_name,
                inputs=[custom_model_name, fm_list],
                outputs=[select_fm_model, fm_list])
        check_status_button.click(fn=trainer.check_if_running, inputs=custom_model_name, outputs=training_status, queue=False)
        check_status_button.click(fn=update_output_files, inputs=custom_model_name, outputs=output_files, queue=False)
        check_status_button.click(fn=update_train_loss_image, inputs=custom_model_name, outputs=train_loss_image, queue=False)
    return demo

model_status = init_model()
# model_status = """模型未加载，请到页面右侧"模型配置"选项卡中"加载模型"按钮"""
pipe = InferencePipeline()
trainer = GLMTrainer()

with gr.Blocks(css=block_css) as demo:
    vs_path, file_status, model_status, vs_list, fm_list = gr.State(""), gr.State(""), gr.State(model_status), gr.State(vs_list), gr.State(fine_tuned_model_list)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",
                                   ).style(container=False)
            with gr.Column(scale=5):
                with gr.Row(scale=5):
                    with gr.Tab("模型配置"):
                        llm_model = gr.Radio(llm_model_dict_list,
                                            label="LLM 模型",
                                            value=LLM_MODEL,
                                            interactive=True)
                        llm_history_len = gr.Slider(0,
                                                    10,
                                                    value=LLM_HISTORY_LEN,
                                                    step=1,
                                                    label="LLM 对话轮数",
                                                    interactive=True)
                        
                        select_fm_model = gr.Dropdown(
                            choices=fm_list.value,
                            label="请选择微调模型",
                            interactive=True,
                            value=LATEST_CHECKPOINT,)

                        use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                                    label="使用p-tuning-v2微调过的模型",
                                                    interactive=True)
                        use_lora = gr.Checkbox(False,
                                                label="使用基于Lora微调过的模型",
                                                interactive=True)
                        embedding_model = gr.Radio(embedding_model_dict_list,
                                                label="Embedding 模型",
                                                value=EMBEDDING_MODEL,
                                                interactive=True)
                        top_k = gr.Slider(1,
                                        20,
                                        value=VECTOR_SEARCH_TOP_K,
                                        step=1,
                                        label="向量匹配 top k",
                                        interactive=True)
                        load_model_button = gr.Button("重新加载模型")
                with gr.Row(scale=5):
                    with gr.Tab("知识库配置"):
                        mode = gr.Radio(["LLM 对话", "知识库问答"],
                                        label="请选择使用模式",
                                        value="LLM 对话", )
                        vs_setting = gr.Accordion("配置知识库", visible=False)
                        mode.change(fn=change_mode,
                                    inputs=mode,
                                    outputs=vs_setting)
                        with vs_setting:
                            select_vs = gr.Dropdown(vs_list.value,
                                                    label="请选择要加载的知识库",
                                                    interactive=True,
                                                    value=vs_list.value[0] if len(vs_list.value) > 0 else None
                                                    )
                            vs_name = gr.Textbox(label="请输入新建知识库名称",
                                                lines=1,
                                                interactive=True)
                            vs_add = gr.Button(value="添加至知识库选项")
                            vs_add.click(fn=add_vs_name,
                                        inputs=[vs_name, vs_list, chatbot],
                                        outputs=[select_vs, vs_list, chatbot])

                            file2vs = gr.Column(visible=False)
                            with file2vs:
                                # load_vs = gr.Button("加载知识库")
                                gr.Markdown("向知识库中添加文件")
                                with gr.Tab("上传文件"):
                                    files = gr.File(label="添加文件",
                                                    file_types=['.txt', '.md', '.docx', '.pdf'],
                                                    file_count="multiple",
                                                    show_label=False
                                                    )
                                    load_file_button = gr.Button("上传文件并加载知识库")
                                with gr.Tab("上传文件夹"):
                                    folder_files = gr.File(label="添加文件",
                                                        # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                        file_count="directory",
                                                        show_label=False
                                                        )
                                    load_folder_button = gr.Button("上传文件夹并加载知识库")
                            # load_vs.click(fn=)
                            select_vs.change(fn=change_vs_name_input,
                                            inputs=select_vs,
                                            outputs=[vs_name, vs_add, file2vs, vs_path])
                            # 将上传的文件保存到content文件夹下,并更新下拉框
                            load_file_button.click(get_vector_store,
                                                show_progress=True,
                                                inputs=[select_vs, files, chatbot],
                                                outputs=[vs_path, files, chatbot],
                                                )
                            load_folder_button.click(get_vector_store,
                                                    show_progress=True,
                                                    inputs=[select_vs, folder_files, chatbot],
                                                    outputs=[vs_path, folder_files, chatbot],
                                                    )
                            query.submit(get_answer,
                                        [query, vs_path, chatbot, mode],
                                        [chatbot, query],
                                    )

    with gr.Tab("知识库管理"):
        vs_setting_a = gr.Accordion("配置知识库")
        with gr.Row():
            with gr.Column(scale=10):
                # 知识点清单
                None
            with gr.Column(scale=5):
                # 选择、添加知识库
                # 上传知识
                with vs_setting_a:
                    select_vs_a = gr.Dropdown(vs_list.value,
                                            label="知识库列表",
                                            interactive=True,
                                            value=vs_list.value[0] if len(vs_list.value) > 0 else None
                                            )
                    vs_name_a = gr.Textbox(label="请输入新建知识库名称",
                                        lines=1,
                                        interactive=True)
                    vs_add_a = gr.Button(value="添加至知识库选项")
                    vs_add_a.click(fn=add_vs_name,
                                inputs=[vs_name_a, vs_list, chatbot],
                                outputs=[select_vs_a, vs_list, chatbot])

                    file2vs_a = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        with gr.Tab("上传文件"):
                            files_a = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button_a = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files_a = gr.File(label="添加文件",
                                                # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                file_count="directory",
                                                show_label=False
                                                )
                            load_folder_button_a = gr.Button("上传文件夹并加载知识库")
                    # load_vs.click(fn=)
                    select_vs_a.change(fn=change_vs_name_input,
                                    inputs=select_vs_a,
                                    outputs=[vs_name_a, vs_add_a, file2vs_a, vs_path])
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    load_file_button_a.click(get_vector_store,
                                        show_progress=True,
                                        inputs=[select_vs_a, files_a, chatbot],
                                        outputs=[vs_path, files_a, chatbot],
                                        )
                    load_folder_button_a.click(get_vector_store,
                                            show_progress=True,
                                            inputs=[select_vs_a, folder_files_a, chatbot],
                                            outputs=[vs_path, folder_files_a, chatbot],
                                            )

    with gr.Tab("模型管理"):
        create_training_demo(trainer, pipe)

    load_model_button.click(reinit_model,
                            show_progress=True,
                            inputs=[llm_model, embedding_model, llm_history_len, use_ptuning_v2, use_lora, top_k, chatbot, select_fm_model],
                            outputs=chatbot
                            )

demo.queue(concurrency_count=3
           ).launch(server_name='0.0.0.0',
                    server_port=7860,
                    show_api=False,
                    share=True,
                    inbrowser=True)
