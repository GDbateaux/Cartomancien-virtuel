import cv2

from pathlib import Path

from card_extractor import CardExtractor
from card_recognizer import CardRecognizer
from tarot_reader import TarotReader
from tts import speak


if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data'
    ref_dir = data_dir / 'cards'
    image_paths = sorted(
        list(data_dir.glob('*.jpg')) +
        list(data_dir.glob('*.png'))
    )

    card_recognizer = CardRecognizer(ref_dir)
    tarot_reader = TarotReader()

    for img_path in image_paths:
        cards = CardExtractor(str(img_path), debug=True).get_cards()
        

        for card in cards:
            best_label, best_score = card_recognizer.recognize(card.image)[0]
            print(best_label, best_score)
    
            card.label = best_label
            card.confidence = best_score

            cv2.imshow('Card', card.image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f'Extracted {len(cards)} cards from {img_path.name}')

    #card_names = [card.label for card in cards]
    card_names = ['le diable', "l'homme pendu", 'la mort']
    reading = tarot_reader.predict(card_names)
    print(reading)
    speak(reading)
