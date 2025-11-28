# Cartomancien-virtuel

## Requirements

To run this project, you need:

- [Python](https://www.python.org/) installed on your system
- [uv](https://docs.astral.sh/uv/) installed (Python package and environment manager)
- [Ollama](https://ollama.com/) installed (for running the local LLM)

## Installation

From the project root:

```bash
# Install all dependencies declared in pyproject.toml
uv sync
ollama pull llama3.2:3b
```

## Sources

This project relies in particular on:

- The official OpenCV documentation: https://docs.opencv.org  
- The official pytorch documentation: https://docs.pytorch.org/docs/stable/index.html
- The official Ollama documentation: https://docs.ollama.com/
- Assistance from ChatGPT (OpenAI) for design discussions, code structure suggestions, and help debugging errors 

The image embedding and similarity approach used in this project was inspired by:

- Tayyib Ul Hassan Gondal, *Image Embeddings for Enhanced Image Search*, The Deep Hub (Medium).  
  https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42  

- PyTorch Hub, *ResNet: Deep Residual Learning for Image Recognition*.  
  https://pytorch.org/hub/pytorch_vision_resnet/
