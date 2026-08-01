"""
Handles compressing a real image into a list of discrete integers (latent codes).
This represents the 'Laptop/Sender' side of our project.
"""
import torch
from PIL import Image
import torchvision.transforms as T
import config
from model_loader import get_vq_model

def prepare_image(image_path):
    """Loads and preprocesses the image for the VQ-VAE."""
    image = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])
    
    # Add a batch dimension: [1, Channels, Height, Width]
    return transform(image).unsqueeze(0)

def encode_to_integers(model, image_tensor):
    """Passes the image through the encoder and extracts the integer codes."""
    print("Encoding image into discrete latent codes...")
    
    with torch.no_grad():
        # 1. Encode into continuous latents
        encoded_output = model.encode(image_tensor)
        latents = encoded_output.latents
        
        # 2. Quantize into discrete integers
        _, _, info = model.quantize(latents)
        
        # info[2] contains the 1D tensor of integer indices
        encoding_indices = info[2]
        
    return encoding_indices

if __name__ == "__main__":
    # 1. Load Model
    model = get_vq_model()
    
    # 2. Prepare Image
    print(f"Reading {config.INPUT_IMAGE_PATH}...")
    img_tensor = prepare_image(config.INPUT_IMAGE_PATH)
    
    # 3. Encode
    latent_integers = encode_to_integers(model, img_tensor)
    
    print(f"Original shape: {img_tensor.shape}")
    print(f"Compressed to {len(latent_integers)} integers.")
    
    # 4. Save the integers to a file (simulating our 'transmission')
    torch.save(latent_integers, config.LATENT_DATA_PATH)
    print(f"Saved latent codes to {config.LATENT_DATA_PATH}")
