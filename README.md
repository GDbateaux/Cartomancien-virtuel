# Cartomancien-virtuel

## Requirements

To run this project, you need:

- [Python](https://www.python.org/) installed on your system
- [uv](https://docs.astral.sh/uv/) installed (Python package and environment manager)
- [Ollama](https://ollama.com/) installed (for running the local LLM)

## Installation

From the project root:

```bash
# Install Python dependencies and create the virtual environment
uv sync

# Download the local LLM used for predictions
ollama pull llama3.2:3b
```

### Downloading TTS voice models (Piper)

Before running the application, you must download at least one French Piper voice model (the default in the code is fr_FR-tom-medium).
```bash
mkdir -p data/voices
cd data/voices
# Download the French voice model used by default (Tom)
uv run python -m piper.download_voices fr_FR-tom-medium

# (Optional) Download another French voice (Siwis) if you want to experiment
uv run python -m piper.download_voices fr_FR-siwis-medium
```

## Tests

To run the tests, use the following command:
```bash
uv run pytest
```

To measure test coverage, run:
```bash
uv run coverage run -m pytest
uv run coverage report
```

To generate an HTML coverage report, run:
```bash
uv run coverage html
```
Then open htmlcov/index.html in your browser to inspect the detailed coverage.

## Sources

This project relies in particular on:

- The official OpenCV documentation: https://docs.opencv.org  
- The official pytorch documentation: https://docs.pytorch.org/docs/stable/index.html
- The official Ollama documentation: https://docs.ollama.com/
- The pytest documentation: https://docs.pytest.org/en/stable/contents.html
- The piper documentation: https://github.com/OHF-Voice/piper1-gpl/blob/main/README.md
- Assistance from ChatGPT (OpenAI) for design discussions, code structure suggestions, and help debugging errors

The image embedding and similarity approach used in this project was inspired by:

- Tayyib Ul Hassan Gondal, *Image Embeddings for Enhanced Image Search*, The Deep Hub (Medium).  
  https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42  

- PyTorch Hub, *ResNet: Deep Residual Learning for Image Recognition*.  
  https://pytorch.org/hub/pytorch_vision_resnet/
