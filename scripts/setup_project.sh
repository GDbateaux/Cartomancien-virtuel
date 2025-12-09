#!/bin/sh

cd "$(dirname "$0")/.."

echo "Installing Python dependencies"
uv sync

echo "Pulling Ollama model"
ollama pull llama3.2:3b

echo "Creating voices directory and downloading voice"
mkdir -p data/voices
cd data/voices
uv run python -m piper.download_voices fr_FR-tom-medium
