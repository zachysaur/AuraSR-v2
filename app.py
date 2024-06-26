import gradio as gr
from gradio_imageslider import ImageSlider
from PIL import Image
import numpy as np
from aura_sr import AuraSR
import torch
import spaces

# Initialize the AuraSR model
aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR")

def process_image(input_image):
    if input_image is None:
        return None

    # Ensure input_image is a numpy array
    input_array = np.array(input_image)

    # Convert to PIL Image for resizing
    pil_image = Image.fromarray(input_array)

    # Resize the longest side to 256 while maintaining aspect ratio
    width, height = pil_image.size
    if width > height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
    
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    # Convert back to numpy array
    resized_array = np.array(resized_image)

    # Upscale the image using AuraSR
    with torch.no_grad():
        upscaled_image = aura_sr.upscale_4x(resized_array)

    # Convert result to numpy array if it's not already
    result_array = np.array(upscaled_image)

    return [input_array, result_array]

with gr.Blocks() as demo:
    gr.Markdown("# Image Upscaler using AuraSR")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy")
            process_btn = gr.Button("Upscale Image")
        with gr.Column(scale=1):
            output_slider = ImageSlider(label="Before / After", type="numpy")

    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=output_slider
    )

demo.launch(debug=True)