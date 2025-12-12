import vosk
import json
import pyaudio
import keyboard
import threading


class STT:
    def __init__(self, model_name='vosk-model-fr-0.22'):
        model = vosk.Model(model_name=model_name)
        rate = 16000
        self.rec = vosk.KaldiRecognizer(model, rate)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8192)
        self.text = ''
        self.thread_up = True

        self.lock = threading.Lock()
        self.event = threading.Event()
        self.t1 = threading.Thread(target=self._listen)
        self.t1.start()
        
    def _listen(self):
        while self.thread_up:
            self.event.wait()
            if not self.thread_up:
                break

            data = self.stream.read(2048, exception_on_overflow=False)
            with self.lock:
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    txt = result['text']
                    if txt:
                        self.text += " " + txt

    def listen(self):
        if not self.event.is_set():
            self.event.set()

    def get_text(self):
        self.event.clear()

        with self.lock:
            final_result = json.loads(self.rec.FinalResult())
        final_result = final_result['text']
        if final_result:
            self.text += (' ' + final_result)

        result = self.text.strip()
        self.text = ''
        return result
    
    def close(self):
        self.thread_up = False
        self.event.set()
        self.t1.join()

        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()

if __name__ == '__main__':
    stt = STT('vosk-model-small-fr-0.22')
    print('start listening')
    keyboard.on_press_key('o', lambda _: stt.listen())
    keyboard.on_release_key('o', lambda _: print(stt.get_text()))
    keyboard.wait('esc')
    stt.close()
