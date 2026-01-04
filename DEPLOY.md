# Deployment Guide

This guide covers deploying CrowdVision to Fly.io and Hugging Face Spaces.

## 📦 Prerequisites

- Git installed
- Account on [Fly.io](https://fly.io) and/or [Hugging Face](https://huggingface.co)

---

## 🚀 Deploy to Fly.io (FastAPI)

### Step 1: Install Fly CLI

**Windows (PowerShell):**
```powershell
powershell -Command "irm https://fly.io/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Login
```bash
flyctl auth login
```

### Step 3: Initialize App
```bash
cd CrowdCounting
flyctl launch --no-deploy
```
- Enter app name (e.g., `crowdvision`)
- Select region (e.g., `sin` for Singapore)
- Skip PostgreSQL and Redis

### Step 4: Deploy
```bash
flyctl deploy
```

### Step 5: Access Your App
```bash
flyctl open
```

Your app will be available at: `https://your-app-name.fly.dev`

### Troubleshooting

**Memory Issues:**
```bash
# Scale up VM
flyctl scale vm shared-cpu-2x --memory 2048
```

**View Logs:**
```bash
flyctl logs
```

---

## 🤗 Deploy to Hugging Face Spaces (Gradio)

### Step 1: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Enter Space name: `crowdvision`
3. Select License: MIT
4. Select SDK: **Gradio**
5. Click "Create Space"

### Step 2: Clone Your Space
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crowdvision
cd crowdvision
```

### Step 3: Copy Files

From your CrowdCounting project, copy:
```bash
# Copy app modules
cp -r ../CrowdCounting/app ./
cp -r ../CrowdCounting/weights ./

# Copy Gradio app
cp ../CrowdCounting/app_gradio.py ./

# Copy HF-specific files
cp ../CrowdCounting/requirements_hf.txt ./requirements.txt
cp ../CrowdCounting/README_HF.md ./README.md
```

### Step 4: Push to Deploy
```bash
git add .
git commit -m "Initial deployment"
git push
```

### Step 5: Wait for Build
- HF Spaces will automatically build and deploy
- This may take 5-10 minutes due to model size
- Check build logs in the Spaces UI

### Enable GPU (Optional)
1. Go to Space Settings
2. Under "Hardware", select GPU (T4 small is free for some accounts)

---

## 📊 Comparison

| Feature | Fly.io (FastAPI) | HF Spaces (Gradio) |
|---------|------------------|-------------------|
| Interface | Custom Web UI | Gradio Default |
| GPU | Paid | Free (limited) |
| Custom Domain | Yes | No |
| Free Tier | Limited | Generous |
| Setup | Medium | Easy |

---

## 🔧 Environment Variables

### Fly.io
Set in `fly.toml` or via CLI:
```bash
flyctl secrets set MY_VAR=value
```

### HF Spaces
Set in Space Settings → Repository Secrets

---

## 📝 Notes

1. **Model Files**: Both deployments require the model weights in `weights/` directory
2. **Memory**: P2PNet requires ~2GB RAM minimum
3. **Cold Starts**: Free tiers may have cold start delays
