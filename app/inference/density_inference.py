"""
Density Map Inference Module

Handles loading CSRNet model and running inference to get crowd count
from density map estimation.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.models.csrnet import load_csrnet

# Global model cache
_model = None
_device = 'cpu'

# Image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_H, TARGET_W = 768, 1024


def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((TARGET_H, TARGET_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model(weights_path: str, quantize: bool = True):
    """Load model into memory (singleton pattern)."""
    global _model, _device
    
    if _model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DensityMap] Loading CSRNet model on {_device}...")
        _model = load_csrnet(weights_path, device=_device, quantize=quantize)
        print(f"[DensityMap] Model loaded successfully!")
    
    return _model


def create_density_visualization(density_map: np.ndarray, count: int) -> str:
    """
    Create a visualization of the density map.
    
    Returns:
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    # Plot density map with jet colormap
    im = ax.imshow(density_map, cmap='jet')
    ax.set_title(f'Density Map | Estimated Count: {count}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def predict_density(image_bytes: io.BytesIO, model=None) -> dict:
    """
    Run density map inference on an image.
    
    Args:
        image_bytes: BytesIO object containing the image
        model: Optional pre-loaded model
    
    Returns:
        dict with 'count', 'density_visualization' (base64), 'method'
    """
    # Load image
    image = Image.open(image_bytes).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Preprocess
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0)
    
    # Get model
    if model is None:
        model = _model
    
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Inference
    with torch.no_grad():
        img_tensor = img_tensor.to(_device)
        density_map = model(img_tensor)
    
    # Get count (sum of density map)
    count = int(density_map.sum().item())
    
    # Create visualization
    density_np = density_map.squeeze().cpu().numpy()
    
    # Upscale density map for better visualization
    density_upscaled = F.interpolate(
        density_map,
        size=(TARGET_H, TARGET_W),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    visualization = create_density_visualization(density_upscaled, count)
    
    return {
        'count': count,
        'method': 'density_map',
        'density_visualization': visualization,
        'original_size': {'width': original_size[0], 'height': original_size[1]}
    }
