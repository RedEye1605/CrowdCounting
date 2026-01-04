# Crowd Counter AI

AI-powered crowd counting web application using deep learning models.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)

## 🎯 Features

- **Density Map Estimation (CSRNet)**: Generates heat map showing crowd distribution
- **Point Localization (P2PNet)**: Detects individual head positions
- **Modern Web Interface**: Clean, bright UI with drag-and-drop upload
- **Quantized Models**: Optimized for CPU deployment with reduced memory

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Docker

```bash
# Build image
docker build -t crowd-counter .

# Run container
docker run -p 8080:8080 crowd-counter
```

## 🌍 Deploy to Fly.io

```bash
# Install flyctl
# Windows: powershell -Command "irm https://fly.io/install.ps1 | iex"

# Login
flyctl auth login

# Deploy
flyctl launch --no-deploy
flyctl deploy
```

## 🤗 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Clone and push your files:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crowd-counter
cd crowd-counter

# Copy files
cp -r app/ weights/ app_gradio.py ./
cp requirements_hf.txt requirements.txt
cp README_HF.md README.md

# Push
git add .
git commit -m "Initial commit"
git push
```

> **Note**: HF Spaces provides free GPU for inference!

## 📁 Project Structure

```
crowd-counting/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── csrnet.py        # CSRNet architecture
│   │   └── p2pnet.py        # P2PNet architecture
│   ├── inference/
│   │   ├── density_inference.py
│   │   └── localization_inference.py
│   ├── static/              # CSS & JS
│   └── templates/           # HTML
├── weights/                 # Model weights (.pth)
├── app_gradio.py            # Gradio app (HF Spaces)
├── requirements.txt         # Fly.io deps
├── requirements_hf.txt      # HF Spaces deps
├── Dockerfile
└── fly.toml
```

## 📊 Models

| Model | Method | Architecture | Size |
|-------|--------|--------------|------|
| CSRNet | Density Map | VGG16 + Dilated Conv | ~65MB |
| P2PNet | Localization | VGG16_bn + FPN | ~86MB |

## 📝 License

MIT License
