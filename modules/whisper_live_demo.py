import threading
import queue
import sounddevice as sd
import numpy as np
import whisper

class WhisperRealtime:
    def __init__(self, model_name="small", language="vi", device="cpu", samplerate=16000, blocksize=4000):
        self.model = whisper.load_model(model_name, device=device)
        self.language = language
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.audio_queue = queue.Queue()
        self.running = False
        self.transcript = ""

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start(self, duration=15):
        self.running = True
        self.transcript = ""
        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype='float32',
            blocksize=self.blocksize,
            callback=self._audio_callback
        )
        with stream:
            print(f"Listening for {duration} seconds...")
            audio_buffer = []
            for _ in range(int(self.samplerate / self.blocksize * duration)):
                block = self.audio_queue.get()
                audio_buffer.append(block)
            audio = np.concatenate(audio_buffer).flatten()
            print("Transcribing...")
            result = self.model.transcribe(audio, language=self.language, fp16=False, task="transcribe", verbose=False, word_timestamps=False)
            self.transcript = result["text"]
            print("Transcript:", self.transcript)
        self.running = False
        return self.transcript

    def get_transcript(self):
        return self.transcript
