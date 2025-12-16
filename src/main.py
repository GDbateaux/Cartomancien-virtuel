from src.tarot_app import TarotApp
from src.utils import load_settings, project_root


# Application entry point: initialize paths, create the TarotApp, load settings, and start the camera loop.
if __name__ == '__main__':
    data_dir = project_root() / 'data'
    ref_dir = data_dir / 'cards'
    tarot_app = TarotApp(ref_dir, 1, 3, 1, model_name_stt='vosk-model-small-fr-0.22')
    
    settings = load_settings()
    if 'camera_index' in settings:
        tarot_app.run(settings['camera_index'])
    else:
        tarot_app.run(0)
