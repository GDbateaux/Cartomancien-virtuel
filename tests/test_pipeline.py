import cv2
import time
import pytest

from pathlib import Path

from src.tarot_app import TarotApp


MAX_TIME_PER_IMAGE = 0.2
DATA_DIR = Path(__file__).parent.parent / 'data'
REF_DIR = DATA_DIR / 'cards'
IMG_TEST_DIR = DATA_DIR / 'img_test'
app = TarotApp(ref_dir=REF_DIR)
app.stt.close()

@pytest.mark.parametrize(
    'filename',
    [
        ('img1.jpg'),
        ('img2.jpg'),
        ('img3.jpg'),
        ('img4.jpg'),
        ('img5.jpg'),
        ('img6.jpg'),
        ('img7.jpg'),
    ],
)
def test_card_pipeline_speed(filename, monkeypatch):
    monkeypatch.setattr(app.tts, 'speak', lambda text: None)
    monkeypatch.setattr(app.tarot_reader, 'stream_predict', lambda cards: [])

    img_path = IMG_TEST_DIR / filename
    image = cv2.imread(img_path)
    num_runs = 10
    total_time = 0.0

    for _ in range(num_runs):
        t0 = time.time()
        app._process_frame(image)
        total_time += time.time() - t0
    mean_time = total_time / num_runs
    print(f'{filename}: {mean_time*1000:.1f} ms')

    assert mean_time < MAX_TIME_PER_IMAGE

