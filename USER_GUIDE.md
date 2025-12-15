# User Guide — Cartomancien Virtuel
## System requirements
### Hardware
- A webcam
- A microphone (for push-to-talk questions)
- Speakers/headphones (for TTS)

### Software

- [Python](https://www.python.org/) installed on your system
- [uv](https://docs.astral.sh/uv/) installed (Python package and environment manager)
- [Ollama](https://ollama.com/) installed (for running the local LLM)

Note: Some models (Ollama model, embedding model, Vosk/Piper data) may require an internet connection the first time to download them, depending on your setup. After that, usage can be fully local.

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

### Manual setup

From the project root:

Install dependencies:
```bash
uv sync
```

Install / pull the local LLM:
```bash
ollama pull llama3.2:3b
```

Download a Piper French voice model (default is fr_FR-tom-medium):
```bash
mkdir -p data/voices
cd data/voices
uv run python -m piper.download_voices fr_FR-tom-medium
```

## Running the application

From the project root:
```bash
uv run python -m src.main
```

A window should open showing the camera feed.
### Settings (camera index)
You can select which camera OpenCV uses by editing:

- `data/settings.json`

Example:
```json
{ "camera_index": 1 }
```
If the file is missing or invalid, the app falls back to the default camera (index 0).

## How to use
### Tarot reading (3-card spread)
1) When the app starts, it speaks a short welcome message.
2) Place exactly 3 cards in front of the camera.
3) Hold them steady until detection is stable.
4) When the three cards are recognized with enough confidence, the app generates a short reading (2–3 sentences) and speaks it aloud.

#### Tips for best detection
- Avoid glare/reflections on glossy cards.
- Put cards on a plain background with good contrast.
- Keep all four corners visible and don’t crop cards out of frame.
- Separate the cards with a gap so they don’t touch or overlap (avoid placing them too close together).

### Ask a question (push-to-talk)
1) Press and hold `o` to start recording your question (STT).
2) Wait a brief moment before speaking, then ask your question clearly.
3) When you’re done, wait a brief moment before releasing `o` to make sure the last words are captured.
4) Release `o` to stop recording and submit the question.
5) The assistant answers using the RAG system (tarot-only constraints) and speaks the reply.

### Quit
Press `q` while the camera window is focused.

## Customizing the card set (reference images)
- The card recognition step uses the reference images stored in: data/cards/
- To use a different deck or update your cards, replace or edit the images in data/cards/ (keep one image per card, and use clear, well-lit pictures).
- Important: the image filename (without the extension) is used as the card label, and this name is what will appear in the prediction/reading (e.g., Le_Bateleur.jpg → label Le_Bateleur).

## Notes about the RAG behavior
- Your RAG index is stored in: data/chroma_tarot/
- Your source text chunks are loaded from: data/tarot_data/*.txt
- To change what the RAG can answer, you can add, remove, or edit the .txt files in data/tarot_data/ (then rebuild the index if needed).

## Prompt customization

You can customize the assistant behavior by editing the prompt text files in `data/prompts/`

Files:

- `questions_system.txt` — system rules for tarot Q&A (RAG).
- `questions_user.txt` — user prompt template for Q&A (must keep {context} and {question}).
- `reading_system.txt` — system rules for the 3-card tarot reading generation.
- `reading_user.txt` — user prompt template for the 3-card reading (must keep {cards_desc}).

## Basic troubleshooting
### Camera window opens but shows nothing / “Cannot open camera”
- Make sure no other app is using the webcam.
- Try another camera index by editing `data/settings.json` (e.g. set "camera_index": 1, then restart the app).

### Cards are not detected (no outlines) or detection is unstable
- Improve lighting and reduce reflections.
- Move the camera slightly farther away so the full cards are visible.
- Use a high-contrast background.
- If you run the project from source, adjust preprocessing in CardExtractor._preprocess() (threshold block size, morphology kernel, etc.).

### Card recognition is wrong or often fails
- If you run the project from source and recognition is too strict, lower the embedding shortlist threshold min_score slightly (e.g. 0.70) in TarotApp._process_frame(). This affects which candidates are sent to SIFT for reranking:
```python
self.card_recognizer.recognize(card.image, min_score=0.75)
```
- Ensure reference images in data/cards/ match your real cards (style, edition, borders).
- Make sure extracted cards are upright/clean: reflections and blur reduce embedding quality.

### The app can’t answer questions (it says “I don’t have enough information”)
If the spoken answer says it doesn’t have enough information, it usually means the RAG retrieval didn’t find an explicit answer in your knowledge base.

- Check that data/tarot_data/ contains relevant .txt files with the information you expect.
- Check that the Chroma index exists and was built successfully in data/chroma_tarot/.
- If you run the project from source, try retrieving more context by increasing n_results in TarotQuestions(..., n_results=4) to 6 or 8.

### Ollama errors (model not found / connection refused)
- Check Ollama is installed and running:
```bash
ollama list
```
- Pull the model again:
```bash
ollama pull llama3.2:3b
```
- If you’re running Ollama in Docker or a different host, make sure your Python client can reach it.

### No sound / TTS doesn’t play
- Verify your system output device is correct (speakers/headphones).

### Speech-to-text doesn’t capture anything
- Confirm your microphone works in another app.

## Folder layout (important paths)
- `data/cards/` — reference card images for recognition
- `data/chroma_tarot/` — persistent ChromaDB index
- `data/prompts/` — editable prompt templates (system/user prompts)
- `data/tarot_data/` — tarot text files used by RAG
- `data/voices/` — Piper voice models
- `data/vosk/` — Vosk STT model folders
- `data/settings.json` — user-editable app settings (e.g., camera_index)
