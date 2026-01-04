"""
CSRNet Model for Crowd Counting via Density Map Estimation
Based on: https://arxiv.org/abs/1802.10062

Architecture:
- Frontend: VGG16 first 23 layers (pre-trained on ImageNet)
- Backend: Dilated convolutions for multi-scale context
- Output: Single-channel density map where sum = crowd count
"""

import torch
import torch.nn as nn
from torchvision import models


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """Create convolutional layers from config."""
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """
    CSRNet for Crowd Counting.
    
    Input: RGB image tensor [B, 3, H, W]
    Output: Density map tensor [B, 1, H/8, W/8]
    """
    
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        
        # Frontend: VGG16 first 23 layers
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if load_weights else None)
        self.frontend = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # Backend: Dilated convolutions
        self.backend = make_layers(
            [512, 512, 512, 256, 128, 64], 
            in_channels=512, 
            dilation=True
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def load_csrnet(weights_path: str, device: str = 'cpu', quantize: bool = True):
    """
    Load CSRNet with trained weights.
    
    Args:
        weights_path: Path to .pth file
        device: 'cpu' or 'cuda'
        quantize: If True, apply dynamic quantization for smaller memory footprint
    
    Returns:
        model: Loaded model in eval mode
    """
    model = CSRNet(load_weights=False)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Apply quantization for reduced memory usage
    if quantize and device == 'cpu':
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )
    
    return model.to(device)
