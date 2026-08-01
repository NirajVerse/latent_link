"""
Configuration settings for the VQ-VAE optical transfer proof of concept.
Keeping these centralized makes it easy to change models or image sizes later.
"""

# Model Settings
# We are using a pre-trained VQ-VAE from the latent diffusion project
MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"
SUBFOLDER = "vqvae"

# Image Processing Settings
IMAGE_SIZE = 256  # We resize to 256x256 for a consistent latent grid

# File Paths
INPUT_IMAGE_PATH = "test_image.jpg"
LATENT_DATA_PATH = "latent_codes.pt"  # Where we save our "LEGO codes"
OUTPUT_IMAGE_PATH = "reconstructed.jpg"

