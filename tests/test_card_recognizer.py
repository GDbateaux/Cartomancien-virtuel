import pytest

from pathlib import Path

from src.card_extractor import CardExtractor
from src.card_recognizer import CardRecognizer


DATA_DIR = Path(__file__).parent.parent / 'data'
IMG_TEST_DIR = DATA_DIR / 'img_test'
card_recognizer = CardRecognizer(DATA_DIR / 'cards')

#TODO Repair
@pytest.mark.parametrize(
    'filename, expected_cards',
    [
        ('img1.jpg', ['p_r', 'co_9']),
        ('img2.jpg', ['t_9', 'co_10', 'co_r', 't_7', 'co_d']),
        ('img3.jpg', ['t_10', 'p_10', 'p_7', 'ca_d']),
        ('img4.jpg', ['t_v']),
        ('img5.jpg', ['t_7', 'ca_a', 'p_v']),
        ('img6.jpg', ['p_v', 'p_d', 'co_a', 't_6']),
        ('img7.jpg', ['p_10', 'ca_7', 'ca_v', 'p_r', 'ca_r']),
    ],
)
def test_card_recognition_test_images(filename, expected_cards: list[str]):
    img_path = IMG_TEST_DIR / filename

    extractor = CardExtractor.from_file(str(img_path))
    cards = extractor.get_cards()
    assert len(cards) == len(expected_cards)

    predicted_labels = []
    for card in cards:
        results = card_recognizer.recognize(card.image)
        label, _ = results[0]
        predicted_labels.append(label)
    predicted_labels.sort()
    expected_cards.sort()
    assert predicted_labels == expected_cards

def test_recognize_min_score():
    img_path = IMG_TEST_DIR / 'img1.jpg'

    extractor = CardExtractor.from_file(str(img_path))
    cards = extractor.get_cards()

    results = card_recognizer.recognize(cards[0].image, top_k=1)
    assert results

    results_high = card_recognizer.recognize(cards[0].image, top_k=1, min_score=2.0)
    assert results_high == []

def test_recognize_top_k():
    img_path = IMG_TEST_DIR / 'img1.jpg'
    top_k = 4

    extractor = CardExtractor.from_file(str(img_path))
    cards = extractor.get_cards()

    results = card_recognizer.recognize(cards[0].image, top_k=top_k)
    assert len(results) == top_k
    