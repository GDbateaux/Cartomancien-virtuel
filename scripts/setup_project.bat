@echo off
setlocal

cd /d "%~dp0.."

echo Installing Python dependencies
uv sync
if errorlevel 1 goto :error

echo Pulling Ollama model llama3.2:3b
ollama pull llama3.2:3b
if errorlevel 1 goto :error

echo Creating voices directory and downloading voice
if not exist data\voices mkdir data\voices
cd data\voices
uv run python -m piper.download_voices fr_FR-tom-medium
if errorlevel 1 goto :error

pause
goto :eof

:error
echo.
echo Setup failed. Check the error messages above.
exit /b 1
