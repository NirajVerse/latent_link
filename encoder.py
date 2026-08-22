"""
Handles compressing a real image into a list of discrete integers (latent codes).
This represents the 'Laptop/Sender' side of our project.
"""
import torch
from PIL import Image
import torchvision.transforms as T
import config
from model_loader import get_vq_model

MULTIPLE_OF = 4  # the VQModel downsamples 4x, so sides must be multiples of 4


def fit_dimensions(width, height, max_side):
    """Scale an image so its longest side is `max_side`, snapping both
    sides down to multiples of `MULTIPLE_OF` (keeps aspect ratio ~intact)."""
    scale = max_side / max(width, height)
    def snap(v):
        return max(MULTIPLE_OF, int(v * scale) // MULTIPLE_OF * MULTIPLE_OF)
    return snap(width), snap(height)


def prepare_image(image_path):
    """Loads and preprocesses the image for the VQ-VAE.

    Returns (tensor, encoded_size=(w, h), original_size=(w, h)). The tensor
    keeps the original aspect ratio instead of being squashed to a square.
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    enc_w, enc_h = fit_dimensions(orig_w, orig_h, config.IMAGE_SIZE)

    transform = T.Compose([
        T.Resize((enc_h, enc_w)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

    # Add a batch dimension: [1, Channels, Height, Width]
    return transform(image).unsqueeze(0), (enc_w, enc_h), (orig_w, orig_h)

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
    img_tensor, enc_size, orig_size = prepare_image(config.INPUT_IMAGE_PATH)
    
    # 3. Encode
    latent_integers = encode_to_integers(model, img_tensor)
    
    print(f"Original shape: {img_tensor.shape}")
    print(f"Encoded at {enc_size[0]}x{enc_size[1]} (original {orig_size[0]}x{orig_size[1]})")
    print(f"Compressed to {len(latent_integers)} integers.")
    
    # 4. Save the integers to a file (simulating our 'transmission')
    torch.save(latent_integers, config.LATENT_DATA_PATH)
    print(f"Saved latent codes to {config.LATENT_DATA_PATH}")
