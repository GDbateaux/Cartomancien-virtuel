from src.tarot_app import TarotApp
from src.utils import project_root


if __name__ == '__main__':
    data_dir = project_root() / 'data'
    ref_dir = data_dir / 'cards'
    tarot_app = TarotApp(ref_dir, 1, 3, 1, model_name_stt='vosk-model-small-fr-0.22')
    tarot_app.run(0)
