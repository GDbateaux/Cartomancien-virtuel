from pathlib import Path

from tarot_app import TarotApp


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'data'
    ref_dir = data_dir / 'cards'
    tarot_app = TarotApp(ref_dir, 1, 3, 1, model_name_stt='vosk-model-small-fr-0.22')
    tarot_app.run(0)
