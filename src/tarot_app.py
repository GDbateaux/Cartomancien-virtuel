import cv2
import time
import keyboard
import threading

from src.card_recognizer import CardRecognizer
from src.tarot_questions import TarotQuestions
from src.card_extractor import CardExtractor
from src.tarot_reader import TarotReader
from src.stt import STT
from src.tts import TTS


class TarotApp:
    """
    Note: The user flow logic and the refactoring into the TarotApp class were designed and refined with help from ChatGPT 
    """
    def __init__(self, ref_dir, stable_seconds = 1.0, num_cards = 3, time_under_three_cards = 1.0, model_name_tts='fr_FR-tom-medium.onnx', model_name_stt='vosk-model-fr-0.22'):
        self.STABLE_SECONDS = stable_seconds
        self.NUM_CARDS = num_cards
        self.TIME_UNDER_THREE_CARDS = time_under_three_cards

        self.card_recognizer = CardRecognizer(ref_dir)
        self.tarot_reader = TarotReader()
        self.tts = TTS(model_name=model_name_tts)
        self.stt = STT(model_name=model_name_stt)
        self.tarot_questions = TarotQuestions()

        self.speaking_finish = True
        self.last_time = time.time()
        self.last_labels_detected = []
        self.reading_done = False
        self.under_three_since = None

        self.audio_lock = threading.Lock()

    def _process_frame(self, frame):
        cards = CardExtractor(frame).get_cards()
        current_labels = []

        for card in cards:
            results = self.card_recognizer.recognize(card.image, min_score=0.75)
            if not results:
                continue
            card.label, card.confidence = results[0]
            current_labels.append(card.label)
            card.draw_on(frame)

        if len(current_labels) == self.NUM_CARDS and self.speaking_finish and not self.reading_done:
            current_sorted = sorted(current_labels)
            last_sorted = sorted(self.last_labels_detected)

            if not self.last_labels_detected or current_sorted != last_sorted:
                self.last_labels_detected = current_sorted.copy()
                self.last_time = time.time()

            def worker(labels):
                with self.audio_lock:
                    for sentence in self.tarot_reader.stream_predict(labels):
                        self.tts.speak(sentence)
                    self.speaking_finish = True
                    self.reading_done = True

            if time.time() - self.last_time > self.STABLE_SECONDS:
                print('start prediction')
                self.speaking_finish = False
                threading.Thread(target=worker, args=(current_labels,)).start()

            self.under_three_since = None
        else:
            if len(current_labels) < self.NUM_CARDS:
                if self.reading_done:
                    if self.under_three_since is None:
                        self.under_three_since = time.time()
                    elif time.time() - self.under_three_since > self.TIME_UNDER_THREE_CARDS:
                        self.reading_done = False
                        self.last_labels_detected = []
                        self.under_three_since = None
        return frame
    
    def _on_release(self):
        question = self.stt.get_text()
        print(question)
        if not self.speaking_finish:
            return
        
        if not question.strip():
            return

        def worker(q):
            with self.audio_lock:
                self.speaking_finish = False
                try:
                    answer = self.tarot_questions.answer(q)
                    self.tts.speak(answer)
                finally:
                    self.speaking_finish = True

        threading.Thread(target=worker, args=(question,)).start()

    def _on_press(self):
        if self.speaking_finish:
            self.stt.listen()
        
    def run(self, camera_index = 0):
        keyboard.on_press_key('o', lambda _: self._on_press())
        keyboard.on_release_key('o', lambda _: self._on_release())

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print('Cannot open camera')
            cap.release()
            cv2.destroyAllWindows()
            self.stt.close()
            return

        self.tts.speak('Bonjour. Tirez trois cartes de tarot et placez-les devant la caméra.')

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            frame = self._process_frame(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stt.close()
