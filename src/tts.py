import numpy as np
import sounddevice as sd

from piper import PiperVoice, SynthesisConfig

from src.utils import project_root


class TTS():
    """
    Offline text-to-speech (TTS) using Piper.

    This class:
    - Loads a local Piper voice model from data/voices/<model_name>
    - Synthesizes speech from text as a stream of audio chunks
    - Plays the audio in real time using sounddevice

    The speak() method is designed to be simple: give it a string, it plays it out loud.
    """

    # Initialize the TTS engine by loading the Piper voice and setting synthesis parameters.
    def __init__(self, model_name='fr_FR-tom-medium.onnx'):
        model_path = project_root() / 'data' / 'voices' / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Voice model '{model_name}' not found at '{model_path}'. ")
        
        self.voice = PiperVoice.load(model_path)
        self.PREDICTION_CONFIG = SynthesisConfig(
            length_scale=1.0,
            noise_scale=0.3,
            noise_w_scale=0.3,
        )

    # Synthesize and play the given text out loud.
    def speak(self, text):
        # This code is inspired by https://noerguerra.com/how-to-read-text-aloud-with-piper-and-python/https://noerguerra.com/how-to-read-text-aloud-with-piper-and-python/
        stream = sd.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16')
        stream.start()

        for chunk in self.voice.synthesize(text, syn_config=self.PREDICTION_CONFIG):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(audio)
        stream.stop()
        stream.close()

if __name__ == '__main__':
    tts = TTS()
    tts.speak("Je vois trois cartes pleines de lumière devant toi. Elles parlent d'opportunités, de rencontres heureuses et d'une belle énergie qui arrive dans ta vie.")
