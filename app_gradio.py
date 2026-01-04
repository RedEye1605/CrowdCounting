"""
Crowd Counter - Hugging Face Spaces Version
Simple Gradio interface for crowd counting with two methods.
"""

import os
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model imports
from app.models.csrnet import load_csrnet
from app.models.p2pnet import load_p2pnet

# Configuration
WEIGHTS_DIR = "weights"
DENSITY_WEIGHTS = os.path.join(WEIGHTS_DIR, "densitymap_model.pth")
LOCALIZATION_WEIGHTS = os.path.join(WEIGHTS_DIR, "p2pnet_model.pth")

# Image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_H, TARGET_W = 768, 1024

# Global models
density_model = None
localization_model = None


def load_models():
    """Load both models on startup."""
    global density_model, localization_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models on {device}...")
    
    if os.path.exists(DENSITY_WEIGHTS):
        density_model = load_csrnet(DENSITY_WEIGHTS, device=device, quantize=(device == 'cpu'))
        print("✅ Density model loaded")
    
    if os.path.exists(LOCALIZATION_WEIGHTS):
        localization_model = load_p2pnet(LOCALIZATION_WEIGHTS, device=device, quantize=(device == 'cpu'))
        print("✅ Localization model loaded")


def predict_density(image: Image.Image):
    """Predict crowd count using density map method."""
    if density_model is None:
        return None, "Density model not loaded"
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((TARGET_H, TARGET_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    device = next(density_model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        density_map = density_model(img_tensor)
    
    count = int(density_map.sum().item())
    
    # Create visualization
    density_upscaled = F.interpolate(
        density_map,
        size=(TARGET_H, TARGET_W),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(density_upscaled, cmap='jet')
    ax.set_title(f'Density Map | Count: {count}', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)
    
    return result_img, f"**Estimated Count: {count} people**"


def point_nms(points, scores, dist_threshold=8.0):
    """Apply Point NMS to remove nearby/duplicate detections."""
    if len(points) == 0:
        return np.array([], dtype=bool)
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(points), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        distances = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        suppressed[distances < dist_threshold] = True
        suppressed[i] = False
    mask = np.zeros(len(points), dtype=bool)
    mask[keep] = True
    return mask


def predict_localization(image: Image.Image, threshold: float = 0.4):
    """Predict crowd count using point localization method."""
    if localization_model is None:
        return None, "Localization model not loaded"
    
    # Resize to make dimensions divisible by 128
    w, h = image.size
    new_w = (w // 128) * 128 if w >= 128 else 128
    new_h = (h // 128) * 128 if h >= 128 else 128
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    img_tensor = transform(resized_image).unsqueeze(0)
    
    device = next(localization_model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = localization_model(img_tensor)
    
    # Get predictions
    pred_logits = outputs['pred_logits']
    pred_points = outputs['pred_points']
    
    scores = F.softmax(pred_logits, dim=-1)[:, :, 1]
    mask = scores[0] > threshold
    filtered_points = pred_points[0][mask].cpu().numpy()
    filtered_scores = scores[0][mask].cpu().numpy()
    
    # Apply Point NMS
    img_diagonal = np.sqrt(new_w**2 + new_h**2)
    nms_dist = max(40.0, img_diagonal * 0.08)
    
    if len(filtered_points) > 0:
        nms_mask = point_nms(filtered_points, filtered_scores, dist_threshold=nms_dist)
        filtered_points = filtered_points[nms_mask]
    
    # Scale points back to original size
    scale_x = w / new_w
    scale_y = h / new_h
    if len(filtered_points) > 0:
        filtered_points[:, 0] *= scale_x
        filtered_points[:, 1] *= scale_y
    
    count = len(filtered_points)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)
    
    if len(filtered_points) > 0:
        ax.scatter(
            filtered_points[:, 0],
            filtered_points[:, 1],
            c='#FF4444',
            s=25,
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5
        )
    
    ax.set_title(f'Detected Heads | Count: {count}', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)
    
    return result_img, f"**Detected: {count} people** (threshold: {threshold})"


def process_image(image, method, threshold):
    """Main processing function."""
    if image is None:
        return None, "Please upload an image"
    
    image = Image.fromarray(image).convert('RGB')
    
    if method == "Density Map":
        return predict_density(image)
    else:
        return predict_localization(image, threshold)


# Load models on startup
load_models()

# Create Gradio interface
with gr.Blocks(
    title="Crowd Counter AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
    )
) as demo:
    gr.Markdown(
        """
        # 👥 Crowd Counter AI
        
        Upload an image to count the number of people using AI-powered analysis.
        
        **Two Methods Available:**
        - **Density Map**: Generates a heat map showing crowd distribution
        - **Point Localization**: Detects individual head positions
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Crowd Image", type="numpy")
            method = gr.Radio(
                choices=["Density Map", "Point Localization"],
                value="Density Map",
                label="Analysis Method"
            )
            threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Detection Threshold (for Point Localization)",
                visible=True
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Result Visualization", type="pil")
            output_count = gr.Markdown(label="Count Result")
    
    analyze_btn.click(
        fn=process_image,
        inputs=[input_image, method, threshold],
        outputs=[output_image, output_count]
    )
    
    gr.Markdown(
        """
        ---
        **Powered by CSRNet & P2PNet** | Built with PyTorch & Gradio
        """
    )

if __name__ == "__main__":
    demo.launch()
