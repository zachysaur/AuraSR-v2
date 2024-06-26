import gradio as gr
from gradio_imageslider import ImageSlider

from PIL import Image
import numpy as np

from aura_sr import AuraSR
import spaces

# Load the AuraSR model
aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR").to("cuda")
    
@spaces.GPU
def process_image(input_image):
    if input_image is None:
        return None
        
    # Resize input image to 256x256
    input_image = Image.fromarray(input_array).resize((256, 256))

    # Upscale the image using AuraSR
    upscaled_image = aura_sr.upscale_4x(input_image)

    # Convert result to numpy array
    result_array = np.array(upscaled_image)

    return [input_array, result_array]

with gr.Blocks() as demo:
    gr.Markdown("# Image Upscaler using AuraSR")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="pil")
            process_btn = gr.Button("Upscale Image")
        with gr.Column(scale=1):
            output_slider = ImageSlider(label="Before / After", type="numpy")

    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=output_slider
    )

demo.launch(debug=True)