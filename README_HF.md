---
title: Crowd Counter AI
emoji: 👥
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.19.2
app_file: app_gradio.py
pinned: false
license: mit
---

# Crowd Counter AI

AI-powered crowd counting using deep learning.

## Features

- **Density Map (CSRNet)**: Heat map visualization
- **Point Localization (P2PNet)**: Individual head detection

## Models

| Model | Method | Architecture |
|-------|--------|--------------|
| CSRNet | Density Map | VGG16 + Dilated Conv |
| P2PNet | Localization | VGG16_bn + FPN |

## Usage

1. Upload a crowd image
2. Select analysis method
3. Click "Analyze"
