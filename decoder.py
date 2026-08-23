"""
Handles reconstructing an image from a list of discrete integers.
This represents the 'Phone/Receiver' side of our project.
"""
import torch
import torchvision.transforms as T
from PIL import Image
import config
from model_loader import get_vq_model
from chafa.canvas import Canvas
from chafa.loader import Loader
import io
from chafa import *

def decode_from_integers(model, encoding_indices, grid_w, grid_h):
    """Reconstructs the image from integer codes on a grid_w x grid_h latent grid."""
    print("Decoding latent codes back into an image...")

    with torch.no_grad():
        # 1. Look up the continuous vectors in the codebook using the indices.
        # get_codebook_entry views them as [B, H, W, C] and returns
        # [B, C, H, W] ready for the decoder.
        quantized_tensors = model.quantize.get_codebook_entry(
            encoding_indices,
            shape=(1, grid_h, grid_w, 3)  # this model's latent_channels = 3
        )

        # 2. Decode back into an image tensor
        decoded_output = model.decode(quantized_tensors)
        reconstructed_tensor = decoded_output.sample

    return reconstructed_tensor

def save_tensor_as_image(tensor, output_path):
    """Converts the tensor back to a standard image and saves it."""
    # Denormalize from [-1, 1] back to [0, 1]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    
    # Convert back to PIL Image
    image = T.ToPILImage()(tensor.squeeze(0))
    image.save(output_path)
    print(f"Saved reconstructed image to {output_path}")


    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    #loader = Loader()
    #loader.load_from_memory(img_bytes)
    loader = Loader(output_path)
    config = CanvasConfig()
    config.width = 512
    config.height = 512


    canvas = Canvas(config)
    #canvas.config.width = 512
   # canvas.config.height = 512
   # canvas.draw_all_pixels(loader) 

    #print(canvas.print().decode("utf-8"))


if __name__ == "__main__":
    # 1. Load Model
    model = get_vq_model()
    
    # 2. Load the "transmitted" integers
    print(f"Loading latent codes from {config.LATENT_DATA_PATH}...")
    try:
        latent_integers = torch.load(config.LATENT_DATA_PATH)
    except FileNotFoundError:
        print("Error: Could not find latent codes. Run encoder.py first!")
        exit(1)
        
    # 3. Decode
    reconstructed_tensor = decode_from_integers(model, latent_integers)
    
    # 4. Save output
    save_tensor_as_image(reconstructed_tensor, config.OUTPUT_IMAGE_PATH)
