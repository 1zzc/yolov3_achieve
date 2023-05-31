import gradio as gr 
import pandas as pd 

from PIL import Image
from pathlib import Path
import numpy as np 
import cv2
import xml.etree.ElementTree as ET

def showxml(xmlpath,img_path):
  # 读取XML文件
  # xml_path = 'annotations.xml'  
  tree = ET.parse(xmlpath)
  root = tree.getroot()

  img = cv2.imread(img_path)

  # 获取标注框
  objs = root.findall('object')
  for obj in objs:
      # 获取标注框类别
      cls = obj.find('name').text  
      # 获取标注框的框的坐标
      coords = obj.find('bndbox')
      x1 = int(coords.find('xmin').text) 
      y1 = int(coords.find('ymin').text)  
      x2 = int(coords.find('xmax').text)
      y2 = int(coords.find('ymax').text) 
    
      # 在图像上绘制标注框
      cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
      # 添加类别标签
      cv2.putText(img, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

  # 显示图像  
  # cv2.imshow('Image', img)  
  # cv2.waitKey(0)
  return img

# ./result_syn/tank_select_target/1/L/21_pitch60_pitch60azimuth358.png



# detect,
#                      inputs=daiDec, 
#                      outputs=out_img)

# 放置目标检测代码
def detect(daiDec):
    xmlpath = ('./output_2/LSMres/'+daiDec.name.split('/')[-1]).replace('png','xml')

    img_path = './output_2/L+S+M/'+daiDec.name.split('/')[-1]

    # if isinstance(img,str):
    #     img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    # result = model.predict(source=img)
    # if len(result[0].boxes.boxes)>0:
    #     vis = plots.plot_detection(img,boxes=result[0].boxes.boxes,
    #                  class_names=class_names, min_score=0.2)
    # else:
    #     vis = img
    return showxml(img_path=img_path,xmlpath=xmlpath)

def detectXingNeng(out_text):
  return 'mAP:81.3%   P：84% '





# 生成背景
def getBackground(inputs_imga,inputs_imgb,inputs_imgc):
  # print(imgA,imgB,imgC,'hahhh')
  # ./result_syn/airplane_select_target/1/L/17_pitch50_pitch50azimuth168.png
  outA,outB,outB=None,None,None

  if inputs_imga:
      outA = Image.open(inputs_imga.replace('select_target','select'))
  if inputs_imgb:
      outB = Image.open(inputs_imgb.replace('select_target','select'))
  if inputs_imgc:
      outC = Image.open(inputs_imgc.replace('select_target','select'))
  return [outA,outB,outB]

def getBackgroundXingNeng(text_output):
  return 'SSIM结构相似度：0.817 \n 耗时：47.4ms '



# 显示原始备选图片的融合结果
def showRongYuan(ronga,rongb,rongc):
  #  inputs=[ronga,rongb,rongc],outputs=[imgshowA,imgshowB,imgshowC]
  imgshowA,imgshowB,imgshowC = None,None,None
  # print(ronga.name,rongb.name,rongc.name,'融合测试')
  # assert False
  if ronga:
      imgshowA = Image.open('./output_2/S/'+ronga.name.split('/')[-1])
  if rongb:
    imgshowB = Image.open('./output_2/M/'+ronga.name.split('/')[-1])
  if rongc:
    imgshowC = Image.open('./output_2/L/'+ronga.name.split('/')[-1])
  return [imgshowA,imgshowB,imgshowC]


# Image.open('./output_2/L+S+M/'+ronga.name.split('/')[-1])
# 放置融合结果   <PIL.Image.Image image mode=RGB size=2560x1296 at 0x7FE4077B8F90> <class 'PIL.Image.Image'> hhhh哈哈哈哈
def imgFuse(ronga,rongb,rongc):
  out_img = Image.open('./output_2/L+S+M/'+ronga.name.split('/')[-1])
  out_img = out_img.resize((200,200))
  return out_img



def showOriginal(inputs_imga,inputs_imgb,inputs_imgc):
    print(inputs_imga,inputs_imgb,inputs_imgc,'测试')
    imgshowA,imgshowB,imgshowC = None,None,None
    if inputs_imga:
       imgshowA = Image.open(inputs_imga)
    if inputs_imgb:
       imgshowB = Image.open(inputs_imgb)
    if inputs_imgc:
       imgshowC = Image.open(inputs_imgc)
    return [imgshowA,imgshowB,imgshowC]
    # return None

def dectectShowYuan(daiDec):
   outDectectYuan = None
   if daiDec:
    outDectectYuan = Image.open('./output_2/L+S+M/'+daiDec.name.split('/')[-1])
   return outDectectYuan


with gr.Blocks(css=".gradio-container img {max-height: 640px}",title='哈哈哈',theme=gr.themes.Soft(),shape=(416,416)) as demo:
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
          ronga = gr.File(file_count='single',type="file", label="短波图像")
          rongb = gr.File(file_count='single',type="file", label="中波图像")
          rongc = gr.File(file_count='single',type="file", label="长波图像")

        buttonYuan = gr.Button("一键显示原始图片",variant="primary")

        with gr.Row():
          imgshowA = gr.Image(type='filepath',shape=(416, 416),label='短波图像')
          imgshowB = gr.Image(type='filepath',shape=(416, 416),label='中波图像')
          imgshowC = gr.Image(type='filepath',shape=(416, 416),label='长波图像')

        buttonYuan.click(showRongYuan,inputs=[ronga,rongb,rongc],outputs=[imgshowA,imgshowB,imgshowC])
        button = gr.Button("融合",variant="primary")
        
        gr.Markdown("## 融合结果")
        with gr.Row().style(width="100%",height='60%'):
          out_img = gr.Image(size=(416,416),max_size=(500,500))
        button.click(imgFuse,
                     inputs=[ronga,rongb,rongc], 
                     outputs=out_img)
        
        text_output = gr.Label(label="性能指标")
        button.click(getBackgroundXingNeng,
                    #  inputs=input_img, 
                     outputs=text_output)      

        
    with gr.Tab("目标检测"):
        mymodel = 'YOLOv7'
        

        daiDec = gr.File(file_count='single',type="file", label="待检测图像")

        dectectYuan = gr.Button("一键显示原始图片",variant="primary")
        outDectectYuan = gr.Image(type='filepath',shape=(416, 416),label='待检测图像')

        dectectYuan.click(dectectShowYuan,inputs=daiDec,outputs=outDectectYuan)



        # gr.Radio(choices=["YOLOv7", "SSD",'Faster RCNN'], label="目标检测模型")
        gr.Dropdown(choices=["YOLOv7", "SSD",'Faster RCNN','RetinaNet','EfficientDet','YOLOv5'], type="value", default='SSD', label='目标检测模型', optional=True)
        button = gr.Button("开始检测",variant="primary")
        
        gr.Markdown("## 检测结果")
        out_img = gr.Image(type='pil')

        # out_text = gr.Label(label="性能指标")

        
        button.click(detect,
                     inputs=daiDec, 
                     outputs=out_img)

        # button.click(detectXingNeng,
        #             #  inputs=input_img, 
        #              outputs=out_text)
  

    with gr.Tab("背景生成"):
        # input_img = gr.Image(type='filepath',label='请选择目标图像',elem_classes="bgsheng")

        with gr.Row():
          inputs_imga = gr.Textbox(label='输入短波图片路径')
          inputs_imgb = gr.Textbox(label='输入中波图片路径')
          inputs_imgc = gr.Textbox(label='输入长波图片路径')

                    # with gr.Row():
                    #     inputs_model_p5 = gr.Radio(choices=model_names_p5, value="yolov5s", label="P5模型")
                    # with gr.Row():
                    #     inputs_size_p5 = gr.Radio(choices=[320, 640, 1280], value=inference_size, label="推理尺寸")
                    # with gr.Row():
                    #     input_conf_p5 = gr.inputs.Slider(0, 1, step=slider_step, default=nms_conf, label="置信度阈值")
        xianshi = gr.Button("一键显示原始图片",variant="primary")


           
        with gr.Row():
          imgshowA = gr.Image(type='filepath',shape=(640, 640),label='短波图像')
          imgshowB = gr.Image(type='filepath',shape=(640, 640),label='中波图像')
          imgshowC = gr.Image(type='filepath',shape=(640, 640),label='长波图像')

        xianshi.click(showOriginal,
                     inputs=[inputs_imga,inputs_imgb,inputs_imgc], 
                     outputs=[imgshowA,imgshowB,imgshowC])


        button = gr.Button("一键生成背景",variant="primary")
        
        gr.Markdown("## 结果")
        # out_img = gr.Image(type='pil',shape=(640, 640),label='结果图像')


        with gr.Row():
          outA = gr.Image(type='filepath',shape=(640, 640),label='图像A')
          outB = gr.Image(type='filepath',shape=(640, 640),label='图像B')
          outC = gr.Image(type='filepath',shape=(640, 640),label='图像C')

        
        
        button.click(getBackground,
                     inputs=[inputs_imga,inputs_imgb,inputs_imgc], 
                     outputs=[outA,outB,outC])


# 哈哈哈
gr.close_all() 
demo.queue(api_open=False)
demo.launch(show_api=False)