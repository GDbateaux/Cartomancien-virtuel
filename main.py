import cv2

from pathlib import Path

from card_extractor import CardExtractor
from card_recognizer import CardRecognizer


if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data'
    ref_dir = data_dir / 'cards'
    image_paths = sorted(
        list(data_dir.glob('*.jpg')) +
        list(data_dir.glob('*.png'))
    )

    for img_path in image_paths:
        cards = CardExtractor(str(img_path)).get_cards()
        card_recognizer = CardRecognizer(ref_dir)

        for card in cards:
            print(card_recognizer.recognize(card))
            cv2.imshow('Card', card)
            cv2.waitKey(0)
        print(f'Extracted {len(cards)} cards from {img_path.name}')
