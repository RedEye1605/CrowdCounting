---
title: CrowdVision - AI Crowd Counting
emoji: 👥
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app_gradio.py
pinned: false
license: mit
---

# CrowdVision - AI Crowd Counting

<div align="center">

**Estimate crowd size using deep learning**

🎯 Two Methods: Density Map (CSRNet) & Point Detection (P2PNet)

</div>

## Usage

1. **Upload** an image containing a crowd
2. **Select** the analysis method:
   - **Density Map**: Shows heat map visualization of crowd density
   - **Point Localization**: Marks individual head positions
3. **Adjust** the detection threshold (for Point Localization)
4. **View** results with count and visualization

## Models

| Model | Method | Best For |
|-------|--------|----------|
| CSRNet | Density Map | Large crowds, density analysis |
| P2PNet | Point Localization | Precise counting, sparse crowds |

## Parameters

- **Threshold (P2PNet)**: Detection confidence (0.4 recommended)
  - Lower = more detections, more false positives
  - Higher = fewer detections, more accurate

## Links

- [GitHub Repository](https://github.com/RedEye1605/CrowdCounting)
- [FastAPI Version (Fly.io)](https://crowdvision.fly.dev)

## License

MIT License
