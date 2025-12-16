# Developer Guide — Cartomancien Virtuel
## Technical architecture
### High-level flow

The application has two main user flows:

#### 3-card reading (computer vision → LLM → TTS)

1) Camera frame is captured with OpenCV.

2) Card extraction (CardExtractor) detects rectangular card contours and applies perspective correction to produce normalized card images.

3) Card recognition (CardRecognizer) uses ResNet18 embeddings to shortlist the closest reference cards in data/cards/, then applies SIFT feature matching to rerank candidates and select the final label.

4) When exactly 3 cards are confidently recognized and stable for a configured duration, the app calls Tarot reading generation (TarotReader, via Ollama chat API).

5) The reading is spoken aloud via Piper TTS (TTS).

#### Push-to-talk question answering (STT → RAG → LLM → TTS)

1) User holds o to record speech (Vosk STT).

2) On key release, recognized text is passed to TarotQuestions.

3) TarotQuestions retrieves context chunks from ChromaDB (TarotRag) and builds a constrained prompt.

4) A local Ollama model answers (tarot-only constraints enforced by the system prompt).

5) Answer is spoken via TTS.

### Threading / concurrency

- The camera loop runs in the main thread.
- Long tasks run in background threads:
    - LLM reading generation (streamed) runs in a worker thread.
    - Question answering runs in a worker thread.

## Code structure

### Project structure
```text
Cartomancien-virtuel/
├─ assets/                 # Logo and documentation images
├─ data/                   # Runtime assets (used by the app)
│  ├─ cards/               # Reference card images (labels = filenames without extension)
│  ├─ cards_test/          # Test images used by the CV pipeline tests
│  ├─ chroma_tarot/        # Persistent ChromaDB index
│  ├─ prompts/             # User-editable prompt templates (system/user)
│  ├─ tarot_data/          # RAG knowledge base (.txt files)
│  ├─ voices/              # Piper voice models (.onnx)
│  ├─ settings.json        # User-editable app settings (e.g., camera_index)
├─ scripts/                # Tarot wikipedia page extraction
├─ src/                    # Python source code
├─ tests/                  # Pytest test suite
├─ .coveragerc             # Coverage configuration
├─ .gitignore              # Git ignore rules
├─ .python-version         # Python version
├─ DEVELOPER_GUIDE.md      # Developer guide (architecture / tests / style)
├─ pyproject.toml          # Project config + dependencies
├─ USER_GUIDE.md           # User guide (install / usage / troubleshooting)
├─ README.md               # Project landing page
└─ uv.lock                 # uv lockfile
```

### Key files
- `src/tarot_app.py`
    - Orchestrates the full app lifecycle (camera loop, recognition pipeline, and audio/thread coordination).
    - Handles keyboard events (o press/release) and decides when to trigger reading or Q&A.

- `src/card_extractor.py`
    - OpenCV preprocessing: grayscale → blur → adaptive threshold → morphology close.
    - Finds contours, filters candidates by area and convex quadrilateral approximation.
    - Applies perspective transform to output normalized card images (fixed width/height).

- `src/card_recognizer.py`
    - Loads a pretrained ResNet18 backbone and turns it into an embedding extractor (removes final classifier).
    - Builds a reference index from data/cards/:
        - normalized embeddings for fast candidate retrieval,
        - SIFT keypoints/descriptors for feature matching.
    - For each extracted card image:
        - selects a shortlist using embedding cosine similarity,
        - reranks that shortlist using SIFT,
        - returns the top matches with SIFT-based scores.

- `src/card.py`
    - Defines the Card class, which stores the card bounding box, the extracted card image, and the predicted label/confidence.
    - Provides draw_on() to overlay the bounding box and label on the camera frame.

- `src/tarot_reader.py`
    - Builds a tarot reading prompt from 3 card labels.
    - Uses Ollama chat API to generate a concise reading.
    - Supports streaming output and yields partial sentences.

- `src/tarot_questions.py`
    - RAG Q&A interface for user questions.
    - Enforces strict constraints: only answer if explicitly supported by retrieved text.

- `src/tarot_rag.py`
    - Builds / loads a Chroma collection (data/chroma_tarot/).
    - Loads text chunks from data/tarot_data/*.txt.
    - Uses an embedding function for retrieval.

- `src/stt.py`
    - Vosk-based offline speech-to-text.

- `src/tts.py`
    - Piper TTS wrapper using sounddevice.
    - Streams generated audio chunks to the output device.

- `src/utils.py`
    - project_root() for supporting both normal execution and PyInstaller “frozen” execution (sys._MEIPASS).
    - load_settings() loads data/settings.json.
    - load_prompt() loads prompt templates from data/prompts/.

## Tests
### Running tests
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

### Test suite overview

- `test_card_extractor.py`
    - Validates number of extracted cards per test image and basic error cases.

- `test_card_recognizer.py`
    - Validates predicted labels on known test images and parameters behavior.

- `test_pipeline_speed.py`
    - Measures average processing time per frame on test images.

- `test_tarot_questions.py`
    - Ensures strict RAG constraints (off-topic and unknown answers return fixed French sentences).

- `test_tarot_rag.py`
    - Validates chunk loading, Chroma query structure, and deterministic retrieval.

- `test_tarot_reader.py`
    - Checks streaming “time to first sentence” and verifies prompt.

Note: Some tests are intentionally designed to be challenging and may not always pass (e.g., difficult images) to highlight current limits and guide future improvements. Performance tests also measure timing and can vary depending on the machine, but they’re useful to track optimization gains over time.
