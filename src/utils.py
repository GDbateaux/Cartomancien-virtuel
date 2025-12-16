import sys
import json

from pathlib import Path


# Default configuration values used when no settings file is found or it is invalid.
DEFAULT_SETTINGS = {
    'camera_index': 0
}

# Dict helper that prevents KeyError in str.format_map by leaving unknown placeholders untouched.
# Code inspired by https://stackoverflow.com/questions/3536303/python-string-format-suppress-silent-keyerror-indexerror
class Default(dict):
    def __missing__(self, key): 
        return key.join("{}")

# Return the project root directory, handling both normal and PyInstaller-frozen execution. 
def project_root():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS).parent
    return Path(__file__).parent.parent

# Load a prompt from a file, or return the default prompt if the file is missing or invalid.
def load_prompt(file_path: Path, default_prompt: str, parameters = None):
    try:
        if file_path.is_file():
            prompt = file_path.read_text(encoding="utf-8").strip()
            if parameters is not None and all('{' + p + '}' in prompt for p in parameters):
                return prompt
    except:
        pass
    return default_prompt.strip()

# Load application settings from data/settings.json, falling back to defaults as needed.
def load_settings():
    settings = project_root() / 'data' / 'settings.json'

    if not settings.is_file():
        return DEFAULT_SETTINGS.copy()

    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return DEFAULT_SETTINGS.copy()
    except:
        return DEFAULT_SETTINGS.copy()

    result_settings = DEFAULT_SETTINGS.copy()

    for key, value in data.items():
        if key in DEFAULT_SETTINGS:
            result_settings[key] = value
    return result_settings
