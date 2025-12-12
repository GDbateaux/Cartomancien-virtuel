<p align="center">
  <img src="assets/logo.png" width="400" alt="Cartomancien Virtuel Logo"><br>
  <sub>Logo generated with DALL·E (OpenAI).</sub>
</p>

# Cartomancien-virtuel
## Description
This application is a local virtual tarot assistant capable of detecting tarot cards from a live camera feed, recognizing them using a computer-vision pipeline, and generating interpretations through a local language model.  
It also includes a RAG system, allowing the user to ask general tarot-related questions and receive grounded, accurate answers.  

## Requirements
To run this project, you need:

- [Python](https://www.python.org/) installed on your system
- [uv](https://docs.astral.sh/uv/) installed (Python package and environment manager)
- [Ollama](https://ollama.com/) installed (for running the local LLM)

## Highlights

- **Card detection** using a custom OpenCV pipeline with contour filtering and perspective correction.
- **Card recognition** powered by a ResNet18 embedding system.
- **Local tarot interpretation** generated offline using a SLM through Ollama.
- **RAG-powered question answering** enabling responses about tarot structure, history, and symbolism.
- **Natural French text-to-speech** via Piper.
- **Automated project setup** with cross-platform scripts that install dependencies, pull LLM models, and download TTS voices.
- **Test suite**, covering card extraction, card recognition, LLM response timing, and the tarot reading pipeline.

## Installation (Automatic)
### Linux / macOS
From the project root:

```bash
./scripts/setup_project.sh
```

### Windows
From the project root, in a terminal (cmd):
```bat
scripts\setup_project.bat
```

These scripts:

- install Python dependencies with uv

- pull the Ollama model llama3.2:3b

- download the Piper French voice model fr_FR-tom-medium into data/voices

## Installation (Manual)
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

## Run
From the project root:
```
uv run python -m src.main
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

## Contributing

Pull requests are welcome.  
To contribute:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Submit a pull request  

## Sources
This project relies in particular on:

- The official OpenCV documentation: https://docs.opencv.org  
- The official Pytorch documentation: https://docs.pytorch.org/docs/stable/index.html
- The official Ollama documentation: https://docs.ollama.com/
- The pytest documentation: https://docs.pytest.org/en/stable/contents.html
- The piper documentation: https://github.com/OHF-Voice/piper1-gpl/blob/main/README.md
- The chroma documentation: https://docs.trychroma.com/docs
- Assistance from ChatGPT (OpenAI) for design discussions, code structure suggestions, documentation writing, and help debugging errors.

The image embedding and similarity approach used in this project was inspired by:

- Tayyib Ul Hassan Gondal, *Image Embeddings for Enhanced Image Search*, The Deep Hub (Medium).  
  https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42  

- PyTorch Hub, *ResNet: Deep Residual Learning for Image Recognition*.  
  https://pytorch.org/hub/pytorch_vision_resnet/

The RAG and question-answering components of this project were inspired by:

- The tutorial “RAG + ChromaDB + Ollama: Python Guide for Beginners”: https://medium.com/@arunpatidar26/rag-chromadb-ollama-python-guide-for-beginners-30857499d0a0

- The official ChromaDB documentation: https://docs.trychroma.com

The SST section was inspired by:

- https://medium.com/@nimritakoul01/offline-speech-to-text-in-python-f5d6454ecd02

- https://stackoverflow.com/questions/3262346/how-to-pause-and-resume-a-thread-using-the-threading-module 

- https://stackoverflow.com/questions/79253154/use-vosk-speech-recognition-with-python?utm_source=chatgpt.com

- https://stackoverflow.com/questions/10525185/python-threading-how-do-i-lock-a-thread

- https://stackoverflow.com/questions/24072790/how-to-detect-key-presses

The texts in data/tarot_data are extracted from Wikipedia (CC BY-SA 4.0 license).
