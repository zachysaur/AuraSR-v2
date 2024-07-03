import gradio as gr
from gradio_imageslider import ImageSlider
from PIL import Image
import numpy as np
from aura_sr import AuraSR
import torch
import spaces

# Force CPU usage
torch.set_default_tensor_type(torch.FloatTensor)

# Override torch.load to always use CPU
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location=torch.device('cpu'))

# Initialize the AuraSR model
aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR")

# Restore original torch.load
torch.load = original_load

def process_image(input_image):
    if input_image is None:
        raise gr.Error("Please provide an image to upscale.")

    # Convert to PIL Image for resizing
    pil_image = Image.fromarray(input_image)

    # Upscale the image using AuraSR
    upscaled_image = process_image_on_gpu(pil_image)

    # Convert result to numpy array if it's not already
    result_array = np.array(upscaled_image)

    return [input_image, result_array]

@spaces.GPU
def process_image_on_gpu(pil_image):
    return aura_sr.upscale_4x(pil_image)
    
title = """<h1 align="center">AuraSR - An open reproduction of the GigaGAN Upscaler from fal.ai</h1>
<p><center>
<a href="https://blog.fal.ai/introducing-aurasr-an-open-reproduction-of-the-gigagan-upscaler-2/" target="_blank">[Blog Post]</a>
<a href="https://huggingface.co/fal-ai/AuraSR" target="_blank">[Model Page]</a>
</center></p>
"""

with gr.Blocks() as demo:
    
    gr.HTML(title)
    
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

    # Add examples
    gr.Examples(
        examples=[
            "image1.png",
            "image3.png"
        ],
        inputs=input_image,
        outputs=output_slider,
        fn=process_image,
        cache_examples=True
    )

demo.launch(debug=True)