# CrowdVision - AI Crowd Counting

<div align="center">

![CrowdVision Logo](logoCrowd.png)

**AI-powered crowd counting using deep learning models**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Live Demo](#) • [Documentation](#-documentation) • [Deploy Guide](#-deployment)

</div>

---

## 🎯 Overview

CrowdVision is an end-to-end web application for crowd counting using state-of-the-art deep learning models. It offers two complementary approaches:

| Method | Model | Description |
|--------|-------|-------------|
| **Density Map** | CSRNet | Generates heat map showing crowd density distribution |
| **Point Localization** | P2PNet | Detects and marks individual head positions |

## ✨ Features

- 🎨 **Modern Web Interface** - Clean, responsive UI with mint green theme
- 📤 **Drag & Drop Upload** - Easy image upload or use sample images
- 🔥 **Dual Detection Methods** - Choose between density map or point localization
- ⚡ **Optimized Models** - Dynamic quantization for efficient CPU inference
- 🎛️ **Adjustable Threshold** - Fine-tune detection sensitivity for P2PNet
- 📊 **Visual Results** - Interactive visualization with download option

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Local Development

```bash
# Clone the repository
git clone https://github.com/RedEye1605/CrowdCounting.git
cd CrowdCounting

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Docker

```bash
# Build image
docker build -t crowdvision .

# Run container
docker run -p 8080:8080 crowdvision
```

## 🌍 Deployment

### Fly.io (Recommended for FastAPI)

```bash
# Install flyctl
# Windows: powershell -Command "irm https://fly.io/install.ps1 | iex"
# macOS/Linux: curl -L https://fly.io/install.sh | sh

# Login and deploy
flyctl auth login
flyctl launch --no-deploy
flyctl deploy
```

### Hugging Face Spaces (Free GPU)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Clone your Space and copy files:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crowdvision
cd crowdvision

# Copy required files
cp -r app/ weights/ app_gradio.py ./
cp requirements_hf.txt requirements.txt
cp README_HF.md README.md

# Push to deploy
git add .
git commit -m "Initial deployment"
git push
```

> 💡 **Tip**: HF Spaces offers free GPU for faster inference!

## 📁 Project Structure

```
CrowdCounting/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── csrnet.py              # CSRNet architecture
│   │   └── p2pnet.py              # P2PNet architecture
│   ├── inference/
│   │   ├── density_inference.py   # Density map inference
│   │   └── localization_inference.py  # Point detection inference
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   ├── images/
│   │   └── samples/               # Sample images for testing
│   └── templates/
│       └── index.html
├── weights/
│   ├── densitymap_model.pth       # CSRNet weights (~65MB)
│   └── p2pnet_model.pth           # P2PNet weights (~86MB)
├── notebooks/                     # Training notebooks
├── app_gradio.py                  # Gradio app for HF Spaces
├── requirements.txt               # Fly.io dependencies
├── requirements_hf.txt            # HF Spaces dependencies
├── Dockerfile                     # Docker configuration
└── fly.toml                       # Fly.io configuration
```

## 📊 Models

### CSRNet (Density Map)
- **Architecture**: VGG16 frontend + Dilated convolutional backend
- **Output**: Density map where sum of pixels = crowd count
- **Best for**: Large crowds, density distribution analysis

### P2PNet (Point Localization)
- **Architecture**: VGG16_bn backbone + FPN decoder + Regression/Classification heads
- **Output**: Point locations of detected heads
- **Best for**: Precise head counting, sparse to medium density crowds
- **Parameters**:
  - Confidence Threshold: 0.4 (40%)
  - NMS Distance: 8% of image diagonal

## 📝 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/predict/density` | Density map prediction |
| POST | `/predict/localization` | Point detection prediction |

### Example Request

```bash
# Density Map
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict/density

# Point Localization
curl -X POST -F "file=@image.jpg" "http://localhost:8000/predict/localization?threshold=0.4"
```

### Response Format

```json
{
  "count": 25,
  "method": "localization",
  "visualization": "base64_encoded_image",
  "points": [
    {"x": 100.5, "y": 200.3, "confidence": 0.42}
  ],
  "threshold": 0.4
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `HOST` | 0.0.0.0 | Server host |

### Tuning P2PNet

- **Lower threshold (0.3)**: More detections, more false positives
- **Higher threshold (0.5)**: Fewer detections, more accurate
- **Recommended**: 0.4 for balanced results

## 📚 Documentation

### Training
The models were trained on custom crowd counting datasets. See the `notebooks/` directory for:
- `densitymap.ipynb` - CSRNet training
- `p2pnet.ipynb` - P2PNet training

### Preprocessing
- Image normalized with ImageNet mean/std
- P2PNet images resized to multiple of 128 pixels
- Point NMS applied to filter duplicate detections

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch) - Density map estimation
- [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) - Point localization
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Gradio](https://gradio.app/) - ML interface
