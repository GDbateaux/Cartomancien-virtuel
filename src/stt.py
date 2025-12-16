import vosk
import json
import pyaudio
import keyboard
import threading

from src.utils import project_root


class STT:
    """
    Offline speech-to-text (STT) using Vosk.

    This class opens the microphone, runs Vosk decoding in a background thread,
    and lets you start/stop capture on demand.

    How it works:
    - A Vosk model is loaded from data/vosk/<model_name> when available, otherwise by name.
    - The listener thread blocks on an Event and only reads audio while the Event is set.
    - Recognized chunks are appended to an internal buffer and returned via get_text().

    Main methods:
    - listen(): start (or resume) recording/decoding
    - get_text(): stop recording and return the transcript collected so far
    - close(): shut everything down (thread + audio stream)

    References:
    Offline Speech to Text in Python: https://medium.com/@nimritakoul01/offline-speech-to-text-in-python-f5d6454ecd02
    How to pause and resume a thread using the threading module?: https://stackoverflow.com/questions/3262346/how-to-pause-and-resume-a-thread-using-the-threading-module
    Use Vosk speech recognition with Python: https://stackoverflow.com/questions/79253154/use-vosk-speech-recognition-with-python?utm_source=chatgpt.com
    Python threading. How do I lock a thread?: https://stackoverflow.com/questions/10525185/python-threading-how-do-i-lock-a-thread
    How to detect key presses?: https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
    """

    # Initialize the Vosk recognizer and the audio input stream, then start the listener thread.
    def __init__(self, model_name='vosk-model-fr-0.22'):
        model_dir = project_root() / 'data' / 'vosk' / model_name

        if not model_dir.exists():
            model = vosk.Model(model_name=model_name)
        else:
            model = vosk.Model(str(model_dir))
        
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

    # Background loop: waits for the event, then reads audio and appends recognized text chunks.
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
                        self.text += ' ' + txt

    # Start or resume listening/decoding.
    def listen(self):
        if not self.event.is_set():
            self.event.set()

    # Stop listening and return the collected transcript.
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
    
    # Close the audio stream and stop the listener thread.
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
