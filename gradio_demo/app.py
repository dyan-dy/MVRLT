if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.curdir)
    import torch
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

import fire
import gradio as gr
from app.gradio_prod3d import create_ui
from app import generator  # 初始化生成逻辑

_TITLE = '''ProdView3D: 多视图生成高质量交互式3D产品展示'''
_DESCRIPTION = '''
* 用户上传多角度产品图 → 自动生成交互式 3D 模型展示（未来支持完整 AI 建模）
* 一键扫码在手机端查看 AR 效果，贴近消费级平台实际应用场景
'''

def launch():
    generator.init_models()  # 如果有模型加载
    with gr.Blocks(
        title=_TITLE,
        theme=gr.themes.Soft(),  # 比 Monochrome 更适合产品类平台调性
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        create_ui()

    demo.queue().launch(share=True)

if __name__ == '__main__':
    fire.Fire(launch)
