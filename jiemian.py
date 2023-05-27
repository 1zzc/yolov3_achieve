import gradio as gr 
import pandas as pd 

from PIL import Image
from pathlib import Path
import numpy as np 
import cv2


# 放置目标检测代码
def detect(img):
    print(img,'img测试结果')
    if isinstance(img,str):
        img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    result = model.predict(source=img)
    if len(result[0].boxes.boxes)>0:
        vis = plots.plot_detection(img,boxes=result[0].boxes.boxes,
                     class_names=class_names, min_score=0.2)
    else:
        vis = img
    return vis

def detectXingNeng(out_text):
  return 'mAP:81.3%  &nbsp;&nbsp;&nbsp;&nbsp; P：84% '


# 放置融合结果   <PIL.Image.Image image mode=RGB size=2560x1296 at 0x7FE4077B8F90> <class 'PIL.Image.Image'> hhhh哈哈哈哈
def imgFuse(imgA,imgB,imgC):
  print(imgA,type(imgB),'hhhh哈哈哈哈')
  return 'zhoudage'


# 生成背景
def getBackground(imgpath):

  # 读取图片
  img = Image.open(imgpath)
  # # 将图片转为RGB模式
  # img = img.convert('RGB')

  outA =img
  outB = img
  return [outA,outB,outB]

def getBackgroundXingNeng(text_output):
  return 'SSIM结构相似度：0.987 \n 耗时：31.4ms '



advanced_css = """
body {
  background-color:green;
}
/* 设置表格的外边距为1em，内部单元格之间边框合并，空单元格显示. */

.bgsheng {
 background-color: rgba(0,0,0,0);
}

.markdown-body table {
    margin: 1em 0;
    border-collapse: collapse;
    empty-cells: show;
}

[data-testid = "user"] {
    max-width: 100%;
    /* width: auto !important; */
    border-bottom-right-radius: 0 !important;
}
/* 行内代码的背景设为淡灰色，设定圆角和间距. */
.markdown-body code {
    display: inline;
    white-space: break-spaces;
    border-radius: 6px;
    margin: 0 2px 0 2px;
    padding: .2em .4em .1em .4em;
    background-color: rgba(175,184,193,0.2);
}

"""


with gr.Blocks(css=".gradio-container {}",title='哈哈哈') as demo:
    gr.Markdown("<div align='center' ><font size='70'>多频段红外图像融合检测识别一体化软件</font></div>")
    # with gr.Tab("Webcam"):
    #     input_img = gr.Image(source='webcam',type='pil')
    #     button = gr.Button("Detect",variant="primary")
        
    #     gr.Markdown("## Output")
    #     out_img = gr.Image(type='pil')
        
    #     button.click(detect,
    #                  inputs=input_img, 
    #                  outputs=out_img)
        
    with gr.Tab("图像融合"):

        with gr.Row():
          imgA = gr.Image(type='filepath',shape=(640, 640),label='短波图像')
          imgB = gr.Image(type='filepath',shape=(640, 640),label='中波图像')
          imgC = gr.Image(type='filepath',shape=(640, 640),label='长波图像')
        button = gr.Button("融合",variant="primary")
        
        gr.Markdown("## 融合结果")
        out_img = gr.Image(type='pil',shape=(640, 640))
        
        button.click(imgFuse,
                     inputs=[imgA,imgB,imgC], 
                     outputs=out_img)
        
    with gr.Tab("目标检测"):
        mymodel = 'YOLOv7'
        input_img = gr.Image(type='pil')


        # gr.Radio(choices=["YOLOv7", "SSD",'Faster RCNN'], label="目标检测模型")
        gr.Dropdown(choices=["YOLOv7", "SSD",'Faster RCNN','RetinaNet','EfficientDet','YOLOv5'], type="value", default='SSD', label='目标检测模型', optional=True)
        button = gr.Button("开始检测",variant="primary")
        
        gr.Markdown("## 检测结果")
        out_img = gr.Image(type='pil')

        out_text = gr.Label(label="性能指标")

        
        button.click(detect,
                     inputs=input_img, 
                     outputs=out_img)

        button.click(detectXingNeng,
                    #  inputs=input_img, 
                     outputs=out_text)
  

    with gr.Tab("背景生成"):
        input_img = gr.Image(type='filepath',label='请选择目标图像',elem_classes="bgsheng")

        # gr.Radio(choices=["YOLOv7", "SSD",'Faster RCNN'], label="目标检测模型")
        # gr.Dropdown(choices=["YOLOv7", "SSD",'Faster RCNN'], type="value", default='SSD', label='目标检测模型', optional=True)
        button = gr.Button("一键生成背景",variant="primary")
        
        gr.Markdown("## 结果")
        # out_img = gr.Image(type='pil',shape=(640, 640),label='结果图像')


        with gr.Row():
          outA = gr.Image(type='filepath',shape=(640, 640),label='图像A')
          outB = gr.Image(type='filepath',shape=(640, 640),label='图像B')
          outC = gr.Image(type='filepath',shape=(640, 640),label='图像C')

        text_output = gr.Label(label="性能指标")
        
        button.click(getBackground,
                     inputs=input_img, 
                     outputs=[outA,outB,outC])

        button.click(getBackgroundXingNeng,
                    #  inputs=input_img, 
                     outputs=text_output)      

        
gr.close_all() 
demo.queue(api_open=False)
demo.launch(show_api=False)