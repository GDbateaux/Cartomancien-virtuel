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
TIME_UNDER_THREE_CARDS = 1

data_dir = Path(__file__).parent.parent / 'data'
ref_dir = data_dir / 'cards'

card_recognizer = CardRecognizer(ref_dir)
tarot_reader = TarotReader()
speaking_finish = True

last_time = time.time()
last_labels_detected = []
reading_done = False
under_three_since = None

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
    
    if len(current_labels) == NUM_CARDS and speaking_finish and not reading_done:
        current_sorted = sorted(current_labels)
        last_labels_detected.sort()

        if not last_labels_detected or any(x != y for x, y in zip(last_labels_detected, current_sorted)):
            last_labels_detected = current_sorted.copy()
            last_time = time.time()

        def worker(labels):
            global speaking_finish, reading_done
            reading = tarot_reader.predict(labels)
            speak(reading)
            speaking_finish = True
            reading_done = True
            
        if time.time() - last_time > STABLE_SECONDS:
            print('start prediction')
            speaking_finish = False
            threading.Thread(target=worker, args=(current_labels,)).start()
        under_three_since = None
    else:
        if len(current_labels) < NUM_CARDS:
            if reading_done:
                if under_three_since is None:
                    under_three_since = time.time()
                elif time.time() - under_three_since > TIME_UNDER_THREE_CARDS:
                    reading_done = False
                    last_labels_detected = []
                    under_three_since = None

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
