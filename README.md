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
- **French text-to-speech** via Piper.
- **French speech-to-text (STT)** for push-to-talk questions via Vosk.
- **Automated project setup** with cross-platform scripts that install dependencies, pull LLM models, and download TTS voices.
- **Test suite**, covering card extraction, card recognition, LLM response timing, and the tarot reading pipeline.

## Documentation
- **User Guide:** see [`USER_GUIDE.md`](USER_GUIDE.md)
- **Developer Guide:** see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)


## Installation

### Installation via Windows Release
1. Download the latest Windows `.zip` from the GitHub **Releases** page.
2. Extract the archive.
3. Install Ollama and download the required model:
```bash
ollama pull llama3.2:3b
```
4. Run CartomancienVirtuel.exe.

This application is fully local, but it still requires Ollama to run the language model.

### Installation (Manual)
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
```bash
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

- [The OpenCV documentation](https://docs.opencv.org/4.x/)
- [The Pytorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [The Ollama documentation](https://docs.ollama.com/)
- [The pytest documentation](https://docs.pytest.org/en/stable/contents.html)
- [The piper documentation](https://github.com/OHF-Voice/piper1-gpl/blob/main/README.md)
- [The chroma documentation](https://docs.trychroma.com/docs)
- [The pyinstaller documentation](https://pyinstaller.org/en/stable/)
- Assistance from [ChatGPT](https://chatgpt.com/) for design discussions, code structure suggestions, documentation writing, and help debugging errors.

The image embedding and similarity approach used in this project was inspired by:

- [Image Embeddings for Enhanced Image Search]( 
  https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42)
- [PyTorch Hub, *ResNet: Deep Residual Learning for Image Recognition*.]( 
  https://pytorch.org/hub/pytorch_vision_resnet/)

The RAG and question-answering components of this project were inspired by:

- [How to Implement RAG with ChromaDB and Ollama: A Python Guide for Beginners](https://medium.com/@arunpatidar26/rag-chromadb-ollama-python-guide-for-beginners-30857499d0a0)

The SST section was inspired by:

- [Offline Speech to Text in Python](https://medium.com/@nimritakoul01/offline-speech-to-text-in-python-f5d6454ecd02)
- [How to pause and resume a thread using the threading module?](https://stackoverflow.com/questions/3262346/how-to-pause-and-resume-a-thread-using-the-threading-module)
- [Use Vosk speech recognition with Python](https://stackoverflow.com/questions/79253154/use-vosk-speech-recognition-with-python?utm_source=chatgpt.com)
- [Python threading. How do I lock a thread?](https://stackoverflow.com/questions/10525185/python-threading-how-do-i-lock-a-thread)
- [Python threading. How do I lock a thread?](https://stackoverflow.com/questions/24072790/how-to-detect-key-presses)

The texts in data/tarot_data are extracted from Wikipedia (CC BY-SA 4.0 license).
