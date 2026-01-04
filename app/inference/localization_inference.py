"""
Localization Inference Module

Handles loading P2PNet model and running inference to get crowd count
via point localization (detecting individual head positions).
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.models.p2pnet import load_p2pnet

# Global model cache
_model = None
_device = 'cpu'

# Image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model(weights_path: str, quantize: bool = True):
    """Load model into memory (singleton pattern)."""
    global _model, _device
    
    if _model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Localization] Loading P2PNet model on {_device}...")
        _model = load_p2pnet(weights_path, device=_device, quantize=quantize)
        print(f"[Localization] Model loaded successfully!")
    
    return _model


def point_nms(points: np.ndarray, scores: np.ndarray, dist_threshold: float = 8.0) -> np.ndarray:
    """
    Apply Non-Maximum Suppression for point detections.
    
    Points that are too close together (within dist_threshold pixels) 
    are suppressed, keeping only the one with highest score.
    
    Args:
        points: Array of shape (N, 2) with x, y coordinates
        scores: Array of shape (N,) with confidence scores
        dist_threshold: Minimum distance between points (pixels)
    
    Returns:
        Boolean mask of points to keep
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Sort by score descending
    order = np.argsort(-scores)
    
    keep = []
    suppressed = np.zeros(len(points), dtype=bool)
    
    for i in order:
        if suppressed[i]:
            continue
        
        keep.append(i)
        
        # Calculate distances to all other points
        distances = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        
        # Suppress points that are too close
        suppressed[distances < dist_threshold] = True
        suppressed[i] = False  # Don't suppress itself
    
    mask = np.zeros(len(points), dtype=bool)
    mask[keep] = True
    return mask


def create_localization_visualization(
    image: Image.Image,
    points: list,
    count: int
) -> str:
    """
    Create a visualization with detected head positions overlaid on the image.
    
    Args:
        image: Original PIL Image
        points: List of (x, y) tuples for detected heads
        count: Total count
    
    Returns:
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Display image
    ax.imshow(image)
    
    # Plot detected points
    if points:
        points_np = np.array(points)
        ax.scatter(
            points_np[:, 0], 
            points_np[:, 1], 
            c='#FF4444',
            s=25,
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5,
            marker='o'
        )
    
    ax.set_title(f'Detected Heads | Count: {count}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def predict_localization(image_bytes: io.BytesIO, model=None, threshold: float = 0.45) -> dict:
    """
    Run localization inference on an image.
    
    Args:
        image_bytes: BytesIO object containing the image
        model: Optional pre-loaded model
        threshold: Confidence threshold for detection (0.0 - 1.0), default 0.4
    
    Returns:
        dict with 'count', 'points', 'visualization' (base64), 'method'
    """
    # Load image
    image = Image.open(image_bytes).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Resize image to dimensions divisible by 8 (required for P2PNet anchor alignment)
    # Also limit max size to avoid memory issues
    MAX_SIZE = 1920
    w, h = original_size
    
    # Scale down if too large
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        w, h = int(w * scale), int(h * scale)
    
    # Make dimensions divisible by 32 for proper feature map alignment
    # Using 32 instead of 128 preserves more image detail
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    new_w = max(new_w, 128)  # Minimum 128 for proper feature extraction
    new_h = max(new_h, 128)
    
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    resize_scale_x = original_size[0] / new_w
    resize_scale_y = original_size[1] / new_h
    
    # Preprocess
    transform = get_transform()
    img_tensor = transform(resized_image).unsqueeze(0)
    
    # Get model
    if model is None:
        model = _model
    
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Inference
    with torch.no_grad():
        img_tensor = img_tensor.to(_device)
        outputs = model(img_tensor)
    
    # Get predictions
    pred_logits = outputs['pred_logits']
    pred_points = outputs['pred_points']
    
    # Apply softmax to get probabilities
    scores = F.softmax(pred_logits, dim=-1)[:, :, 1]  # Probability of being a person
    
    # Filter by threshold
    mask = scores[0] > threshold
    filtered_points = pred_points[0][mask].cpu().numpy()
    filtered_scores = scores[0][mask].cpu().numpy()
    
    # Apply Point NMS to remove duplicate/nearby detections
    # Use distance threshold based on image size (5% of diagonal - tuned via grid search)
    img_diagonal = np.sqrt(new_w**2 + new_h**2)
    nms_dist = max(15.0, img_diagonal * 0.05)  # Minimum 15 pixels
    
    if len(filtered_points) > 0:
        nms_mask = point_nms(filtered_points, filtered_scores, dist_threshold=nms_dist)
        filtered_points = filtered_points[nms_mask]
        filtered_scores = filtered_scores[nms_mask]
    
    # Convert to list of (x, y) tuples and scale back to original image size
    # Filter out points that are too close to the border (likely model artifacts)
    points = []
    margin = 3  # pixels
    for i, (x, y) in enumerate(filtered_points):
        # Scale points back to original image size
        scaled_x = x * resize_scale_x
        scaled_y = y * resize_scale_y
        
        # Check if point is within image bounds with a small margin
        if margin <= scaled_x <= original_size[0] - margin and \
           margin <= scaled_y <= original_size[1] - margin:
            points.append({
                'x': float(scaled_x),
                'y': float(scaled_y),
                'confidence': float(filtered_scores[i])
            })

    
    count = len(points)
    
    # Create visualization
    point_coords = [(p['x'], p['y']) for p in points]
    visualization = create_localization_visualization(image, point_coords, count)
    
    return {
        'count': count,
        'method': 'localization',
        'points': points,
        'visualization': visualization,
        'original_size': {'width': original_size[0], 'height': original_size[1]},
        'threshold': threshold
    }
