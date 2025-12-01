import threading
import time
import cv2

from pathlib import Path

from card_extractor import CardExtractor
from card_recognizer import CardRecognizer
from tarot_reader import TarotReader
from tts import speak


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cannot open camera')
    exit()

STABLE_SECONDS = 1
NUM_CARDS = 3
data_dir = Path(__file__).parent.parent / 'data'
ref_dir = data_dir / 'cards'
card_recognizer = CardRecognizer(ref_dir)
tarot_reader = TarotReader()
speaking_finish = True

speak("Bonjour. Tirez trois cartes de tarot et placez-les devant la caméra.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    cards = CardExtractor(frame).get_cards()
    current_labels = []

    for card in cards:
        results = card_recognizer.recognize(card.image, min_score=0.75)
        if not results:
            continue
        card.label, card.confidence = results[0]
        current_labels.append(card.label)
        card.draw_on(frame)
    
    if len(current_labels) == NUM_CARDS and speaking_finish is True:
        def worker(labels):
            global speaking_finish
            reading = tarot_reader.predict(labels)
            speak(reading)
            speaking_finish = True
            
        current_labels.sort()
        speaking_finish = False
        threading.Thread(target=worker, args=(current_labels,)).start()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

""" if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'data'
    ref_dir = data_dir / 'cards'
    image_paths = sorted(
        list(data_dir.glob('*.jpg')) +
        list(data_dir.glob('*.png'))
    )

    card_recognizer = CardRecognizer(ref_dir)
    tarot_reader = TarotReader()

    for img_path in image_paths:
        cards = CardExtractor.from_file(str(img_path)).get_cards()
        

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
    speak(reading) """
