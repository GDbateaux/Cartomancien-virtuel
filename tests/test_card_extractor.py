import pytest

from pathlib import Path

from src.card_extractor import CardExtractor


DATA_DIR = Path(__file__).parent.parent / 'data'
IMG_TEST_DIR = DATA_DIR / 'img_test'

@pytest.mark.parametrize(
    'filename, expected_count',
    [
        ('img1.jpg', 2),
        ('img2.jpg', 5),
        ('img3.jpg', 4),
        ('img4.jpg', 1),
        ('img5.jpg', 3),
        ('img6.jpg', 4),
        ('img7.jpg', 5),
    ],
)
def test_number_cards_extracted(filename, expected_count):
    cards = CardExtractor.from_file(IMG_TEST_DIR / filename).get_cards()
    assert len(cards) == expected_count

def test_init_raises_on_none_image():
    with pytest.raises(ValueError):
        CardExtractor(img=None)

def test_from_file_raises_on_missing_file():
    img_path = IMG_TEST_DIR / "does_not_exist.jpg"
    with pytest.raises(ValueError):
        CardExtractor.from_file(str(img_path))
