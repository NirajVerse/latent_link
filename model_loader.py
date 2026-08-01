"""
Handles loading the pre-trained VQ-VAE model from Hugging Face.
"""
import torch
from diffusers import VQModel
import config

def get_vq_model():
    """
    Loads and returns the VQModel in evaluation mode.
    """
    print(f"Loading VQModel: {config.MODEL_ID}...")
    
    # Load the model from Hugging Face
    model = VQModel.from_pretrained(config.MODEL_ID, subfolder=config.SUBFOLDER)
    
    # Set to evaluation mode (important since we aren't training)
    model.eval()
    
    return model
