"""
P2PNet Model for Crowd Counting via Point Localization
Based on: https://arxiv.org/abs/2107.12746

Architecture:
- Backbone: VGG16 with BatchNorm (pre-trained on ImageNet)
- FPN Decoder: Multi-scale feature fusion
- Heads: Classification + Regression for anchor-based detection
- Output: Point coordinates of each person's head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import List, Optional


class VGG(nn.Module):
    """VGG network for feature extraction."""
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """Create VGG layers from config."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# VGG16 config
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class BackboneBase_VGG(nn.Module):
    """VGG backbone with intermediate layer outputs."""
    
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []
        if self.return_interm_layers:
            xs = tensor_list
            for layer in [self.body1, self.body2, self.body3, self.body4]:
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """VGG16 backbone with pretrained weights."""
    
    def __init__(self, name: str = 'vgg16_bn', return_interm_layers: bool = True):
        if name == 'vgg16_bn':
            backbone = torchvision.models.vgg16_bn(weights="IMAGENET1K_V1")
        elif name == 'vgg16':
            backbone = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown backbone: {name}")
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class Decoder(nn.Module):
    """FPN-style decoder for multi-scale feature fusion."""
    
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super().__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = self.P4_2(P5_upsampled_x + P4_x)
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = self.P3_2(P4_upsampled_x + P3_x)
        return P3_x, P4_x, P5_x


class RegressionModel(nn.Module):
    """Head for predicting point offsets."""
    
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x.contiguous().view(x.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    """Head for predicting point confidence."""
    
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=2, feature_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        b, w, h, _ = x.shape
        x = x.view(b, w, h, self.num_anchor_points, self.num_classes)
        return x.contiguous().view(b, -1, self.num_classes)


def generate_anchor_points(stride=16, row=3, line=3):
    """Generate anchor points for a given stride."""
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return np.vstack((shift_x.ravel(), shift_y.ravel())).T


def shift(shape, stride, anchor_points):
    """Shift anchor points across the feature map."""
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).T
    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((K, 1, 2)))
    return all_anchor_points.reshape((K * A, 2))


class AnchorPoints(nn.Module):
    """Generate anchor points for the input image."""
    
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super().__init__()
        self.pyramid_levels = pyramid_levels or [3]
        self.strides = strides or [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for i, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, self.row, self.line)
            shifted = shift(image_shapes[i], self.strides[i], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32))


class P2PNet(nn.Module):
    """
    P2PNet for Crowd Counting via Point Localization.
    
    Input: RGB image tensor [B, 3, H, W]
    Output: Dictionary with 'pred_logits' and 'pred_points'
    """
    
    def __init__(self, backbone=None, row=2, line=2):
        super().__init__()
        self.backbone = backbone if backbone else Backbone_VGG('vgg16_bn', True)
        self.num_classes = 2
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, num_classes=self.num_classes, num_anchor_points=num_anchor_points)
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples):
        features = self.backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]

        regression = self.regression(features_fpn[1]) * 100
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        
        # Move anchor points to same device as input
        anchor_points = anchor_points.to(samples.device)

        output_coord = regression + anchor_points
        output_class = classification
        return {'pred_logits': output_class, 'pred_points': output_coord}


def load_p2pnet(weights_path: str, device: str = 'cpu', quantize: bool = True):
    """
    Load P2PNet with trained weights.
    
    Args:
        weights_path: Path to .pth file
        device: 'cpu' or 'cuda'
        quantize: If True, apply dynamic quantization for smaller memory footprint
    
    Returns:
        model: Loaded model in eval mode
    """
    model = P2PNet(row=2, line=2)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle checkpoint format: could be {'model': state_dict} or just state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
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
