<div align="center">
  <a href="http://neuralnet.solutions" target="_blank">
    <img width="450" src="https://raw.githubusercontent.com/NeuralNet-Hub/assets/main/logo/LOGO_png_orig.png">
  </a>
</div>

# ALPR YOLO Models by [NeuralNet](https://neuralnet.solutions) 🚗

This repository provides **open-source Automatic License Plate Recognition (ALPR) models** built on the YOLO11 architecture. Our models are designed for high-performance license plate detection and recognition across various real-world scenarios.

![ALPR Demo](assets/demo.gif)

The complete solution includes:
- **High-accuracy YOLO11-based ALPR models** for license plate detection
- **Gradio web interface** for easy testing and deployment  
- **Ollama integration** for OCR text extraction from detected plates
- **Docker support** with GPU acceleration
- **Production-ready** inference pipeline

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Docker (optional, for containerized deployment)
- Ollama server for OCR functionality

## Quick Start

1. **Download the model weights:**
```bash
# Download from our latest release
wget https://github.com/NeuralNet-Hub/assets/releases/download/v0.0.1/alpr-yolo11s-aug.pt
```

2. **Clone the inference repository:**
```bash
git clone https://github.com/NeuralNet-Hub/alpr-yolo-gradio
cd alpr-yolo-gradio
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python main.py --model alpr-yolo11s-aug.pt
```

## Model Performance

Our ALPR models have been extensively trained and validated to provide:

- **🎯 High Accuracy**: Robust detection across various lighting conditions and angles
- **⚡ Fast Inference**: Optimized for real-time applications  
- **🌐 Versatile**: Works with different vehicle types and license plate formats
- **📱 Easy Integration**: Compatible with existing Ultralytics workflows

### Training Metrics
All training data, metrics, and experimental results are available at:
**[WandB Project: ALPR-by-hdnh2006](https://wandb.ai/hdnh2006/ALPR-by-hdnh2006/)**

## Docker Deployment

### Build and Run with GPU Support
```bash
# Build the image
docker build -t alpr-yolo .

# Run with GPU support
docker run --gpus all -p 7860:7860 alpr-yolo

# Run with host network (for local Ollama server)
docker run --gpus all --network host alpr-yolo
```

## Usage Examples

### Web Interface
Access the Gradio interface at `http://localhost:7860` to:
- Upload images for license plate detection
- Adjust confidence and IoU thresholds
- Configure Ollama server for OCR
- View detected plates with extracted text

## Configuration

### Ollama Server Setup
The application integrates with Ollama for OCR functionality:

1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai)
2. **Pull a vision model**: `ollama pull qwen2.5vl:3b`
3. **Configure server URL** in the Gradio interface

### Model Selection
Available pre-trained models:
- `alpr-yolo11s-aug.pt` - Small model, fast inference
- Custom models supported via the web interface

## File Structure

```
.
├── main.py                 # Main application with Gradio interface
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── alpr-yolo11s-aug.pt   # Pre-trained ALPR model
└── README.md             # This file
```

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce image size or batch size
- Use CPU inference: `model.predict(device='cpu')`

**Ollama connection failed:**
- Verify Ollama server is running
- Check server URL configuration
- Ensure vision model is installed

**Poor detection accuracy:**
- Adjust confidence threshold (try 0.1-0.5)
- Ensure good image quality and lighting
- Consider using larger model variants

## License

This project is released under the **[Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)**, ensuring compatibility with the YOLO ecosystem while maintaining open-source accessibility.

## About NeuralNet

[NeuralNet](https://neuralnet.solutions) is an AI software development company specializing in privacy-focused artificial intelligence solutions. Our flagship products include [PrivateGPT](https://neuralnet.solutions/privategpt), a secure and private alternative to traditional AI chatbots that keeps your data completely confidential and runs locally on your infrastructure.

We develop cutting-edge AI tools that prioritize:
- **Data Privacy** - Your information never leaves your environment
- **Enterprise Security** - Military-grade encryption and access controls  
- **Easy Deployment** - Ready-to-use solutions with minimal setup
- **Open Source** - Transparent, community-driven development

### Connect with Us

🌐 **Company Website**: [neuralnet.solutions](https://neuralnet.solutions)  
📝 **Author's Blog**: [henrynavarro.org](https://henrynavarro.org)  
🔒 **Try PrivateGPT**: [chat.privategpt.es](https://chat.privategpt.es)  
📊 **Model Training Data**: [WandB ALPR Project](https://wandb.ai/hdnh2006/ALPR-by-hdnh2006/)

---

_Developed by **[Henry Navarro](https://github.com/hdnh2006)** - Building the future of computer vision, one model at a time._
