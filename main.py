#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 16:40:44 2025

@author: henry
"""


import argparse
import base64
import json
import gradio as gr
import numpy as np
import PIL.Image as Image
from ultralytics import ASSETS, YOLO
from ultralytics.utils.plotting import save_one_box, Annotator
from ollama import Client
from io import BytesIO

# Parse command line arguments
parser = argparse.ArgumentParser(description='YOLO Model Configuration')
parser.add_argument('--model', type=str, default='alpr-yolo11s-aug.pt', help='YOLO model path.')
args = parser.parse_args()

# Set model from command line argument
model = YOLO(args.model)

def predict(img, conf_threshold, iou_threshold, ollama_server, model_choice, custom_model):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    # Use custom model if "Custom" is selected, otherwise use the dropdown choice
    ollama_model = custom_model if model_choice == "Custom" else model_choice
    print("Pulling ollama model, this can take several minutes depending on your internet connection")
    
    client = Client(host=ollama_server)
    client.pull(model=ollama_model)
    
    # Getting predictions and results. This code only works for bs=1
    results = predict_image(img, conf_threshold, iou_threshold)[0]
    cropped_images = extract_bounding_boxes(img, results)
        
    response = client.chat(model=ollama_model, messages=[
        {
            'role': 'system',
            'content': 'You are an expert OCR model for license plate'
        },                
        {
            'role': 'user',
            'content': """I need that for each image you provide a single json with the license plate text you read, if you don't see any text, just provide an empty json.
The structure must be ALWAYS like following:
```json
{
"content": [full_plate_1, full_plate_2, full_plate_3]
}
```
Where full_text_image_i is all the text you see per each image, DO NOT split the text of the same image in a list. PUT ALL TEXT WITHOUT SPACE, DO NOT SPLIT text for the same image.

If there's just a single image, the output will be:
```json
{
"content": [full_text_image_1]
}
```
""",
            'images': cropped_images
        },
    ])
    
    
    try:
        text_extracted = json.loads(response.message.content.split('```json\n')[1].split('\n```')[0])
        print("Text extracted by model:")
        print(text_extracted)
    except:
        text_extracted = {'content': ['']*len(results.boxes.xyxy)}
        print("No text extracted")
        
    
    # Suppose pil_img is your PIL image and box = [x1, y1, x2, y2]
    annotator = Annotator(img, pil=True)
    for i, box in enumerate(results.boxes.xyxy):       
        annotator.box_label(box, label=text_extracted["content"][i], color=(255, 0, 0))
    im = annotator.im  # This will be a PIL image with boxes drawn
    
    return im

def pil_to_base64(pil_image, format='JPEG'):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def extract_bounding_boxes(img, results):
    """Extract cropped images from bounding boxes detected by YOLO"""
    cropped_images = []
    boxes = results.boxes
    if boxes is not None:
        for box in boxes:
            # Get bounding box coordinates (xyxy format)
            np_img = np.array(img)
            cropped_img = save_one_box(box.xyxy, np_img, save=False)
            pil_img = Image.fromarray(cropped_img[..., ::-1])
            cropped_images.append(pil_to_base64(pil_img))
    return cropped_images

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )
    return results

def update_custom_visibility(choice):
    """Show or hide the custom model textbox based on dropdown selection"""
    if choice == "Custom":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Using gr.Blocks with proper layout
with gr.Blocks(title="Ultralytics Gradio with Ollama Vision Models") as demo:
    gr.Markdown("""
# Ultralytics Gradio with Ollama OCR by [NeuralNet](https://neuralnet.solutions/)

<div align="left">
  <a href="https://neuralnet.solutions" target="_blank">
    <img width="300" src="https://raw.githubusercontent.com/NeuralNet-Hub/assets/main/logo/LOGO_png_orig.png">
  </a>
</div>

Upload images for OCR inference and License Plate Recognition. The system will detect objects and extract text from them.

*Based on the [Ultralytics Gradio tutorial](https://docs.ultralytics.com/integrations/gradio/) | Developed by **[Henry Navarro](https://github.com/hdnh2006)** from [NeuralNet](https://neuralnet.solutions/)*
""")
    
    with gr.Row():
        # Left column - Input
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Image")
            
            with gr.Row():
                conf_slider = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                iou_slider = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
            
            ollama_server_box = gr.Textbox(
                label="Ollama Server",
                placeholder="Enter Ollama server URL...",
                value="http://192.168.100.60:11434",
                lines=1
            )
            
            model_dropdown = gr.Dropdown(
                choices=["qwen2.5vl:3b", "qwen2.5vl:7b", "llava:7b", "llava:13b", "llava:34b", "Custom"],
                value="qwen2.5vl:3b",
                label="Ollama Vision Model",
                info="Select a model or choose 'Custom' to enter your own. Available models: https://ollama.com/search?c=vision&o=newest"
            )
            
            custom_model_box = gr.Textbox(
                label="Custom Model Name",
                placeholder="Enter your custom model name (e.g., my-model:latest)",
                visible=False,  # Initially hidden
                interactive=True
            )
            
            submit_btn = gr.Button("Process Image", variant="primary")
        
        # Right column - Output
        with gr.Column():
            output_img = gr.Image(type="pil", label="Result")
    
    # Connect the dropdown change to show/hide custom textbox
    model_dropdown.change(
        fn=update_custom_visibility,
        inputs=model_dropdown,
        outputs=custom_model_box
    )
    
    submit_btn.click(
        fn=predict,
        inputs=[img_input, conf_slider, iou_slider, ollama_server_box, model_dropdown, custom_model_box],
        outputs=output_img
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["assets/roboflowimage1.jpg", 0.25, 0.45, "http://192.168.100.60:11434", "qwen2.5vl:3b", ""],
            ["assets/roboflowimage2.jpg", 0.25, 0.45, "http://192.168.100.60:11434", "qwen2.5vl:3b", ""],
        ],
        inputs=[img_input, conf_slider, iou_slider, ollama_server_box, model_dropdown, custom_model_box]
    )

if __name__ == "__main__":
    demo.launch()
