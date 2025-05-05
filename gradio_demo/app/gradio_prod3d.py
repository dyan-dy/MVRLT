import os
import gradio as gr
from app.utils import clean_up
from app.model_runner import generate_3d_model_from_views  # 你需要定义这个函数逻辑

def create_ui(concurrency_id="prodview"):
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 上传多视图产品图像")
            input_images = gr.File(
                label="上传多张产品图片（支持 JPG/PNG）",
                file_types=["image"],
                file_count="multiple",
                type="file"
            )
            
            # 示例图片路径
            example_folder = os.path.join(os.path.dirname(__file__), "../examples")
            if os.path.exists(example_folder):
                example_files = sorted([
                    [os.path.join(example_folder, f)] for f in os.listdir(example_folder) if f.endswith((".jpg", ".png"))
                ])
                gr.Examples(
                    examples=example_files,
                    inputs=[input_images],
                    label="点击示例图开始",
                    cache_examples=False
                )

        with gr.Column(scale=3):
            output_mesh = gr.Model3D(label="生成的3D模型", show_label=True, height=300)
            output_video = gr.Video(label="旋转预览（可选）", visible=True, height=300)

            with gr.Row():
                remove_bg = gr.Checkbox(value=True, label="自动去背景")
                refine_details = gr.Checkbox(value=True, label="精细化建模")
            
            with gr.Row():
                expansion = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, label="扩展权重")
                generate_video = gr.Checkbox(value=True, label="生成旋转视频")

            seed = gr.Slider(minimum=0, maximum=1000000, step=1, value=42, label="随机种子")

            gen_btn = gr.Button("生成3D模型")

    gen_btn.click(
        fn=generate_3d_model_from_views,
        inputs=[input_images, remove_bg, refine_details, expansion, generate_video, seed],
        outputs=[output_mesh, output_video],
        concurrency_id=concurrency_id,
        api_name="generate_product3d",
    ).success(clean_up, api_name=False)

    return input_images
