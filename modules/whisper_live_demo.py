import threading
import queue
import sounddevice as sd
import numpy as np
import whisper

class WhisperRealtime:
    def __init__(self, model_name="large", language="vi", device=None, samplerate=16000, blocksize=4000, chunk_duration=4, chunk_overlap=1):
        # Use GPU if available, else fallback to CPU
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=device)
        self.language = language
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.chunk_duration = chunk_duration  # seconds per chunk
        self.chunk_overlap = chunk_overlap    # seconds overlap
        self.audio_queue = queue.Queue()
        self.running = False
        self.transcript = ""
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def _recognition_worker(self, device=None):
        # Auto-detect input channels for device
        try:
            if device is not None:
                dev_info = sd.query_devices(device)
            else:
                dev_info = sd.query_devices(sd.default.device[0])
            num_channels = min(dev_info['max_input_channels'], 2) if dev_info['max_input_channels'] > 0 else 1
        except Exception as e:
            print(f"[WhisperRealtime] Device info error: {e}, fallback to 1 channel")
            num_channels = 1

        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=num_channels,
            dtype='float32',
            blocksize=self.blocksize,
            callback=self._audio_callback,
            device=device
        )
        with stream:
            buffer = []
            samples_per_chunk = int(self.samplerate * self.chunk_duration)
            samples_overlap = int(self.samplerate * self.chunk_overlap)
            print(f"Listening (realtime)...")
            last_transcribed = 0
            while not self._stop_event.is_set():
                try:
                    block = self.audio_queue.get(timeout=0.5)
                    # Always convert to mono for Whisper
                    if block.ndim == 2 and block.shape[1] > 1:
                        block = block.mean(axis=1).astype(np.float32)
                    else:
                        block = block.flatten().astype(np.float32)
                    buffer.append(block)
                    flat = np.concatenate(buffer).flatten() if buffer else np.array([], dtype=np.float32)
                    while len(flat) - last_transcribed >= samples_per_chunk:
                        start = last_transcribed
                        end = start + samples_per_chunk
                        audio_chunk = flat[start:end]
                        result = self.model.transcribe(audio_chunk, language=self.language, fp16=False, task="transcribe", verbose=False, word_timestamps=False)
                        text = result["text"].strip()
                        if text:
                            from datetime import datetime
                            timestamp = datetime.now().strftime("[%H:%M]")
                            line = f"{timestamp} {text}"
                            with self._lock:
                                self.transcript += ("\n" if self.transcript else "") + line
                        last_transcribed += samples_per_chunk - samples_overlap
                    # Remove old samples to keep buffer size reasonable
                    if last_transcribed > 0 and last_transcribed > samples_per_chunk:
                        flat = flat[last_transcribed:]
                        buffer = [flat]
                        last_transcribed = 0
                except queue.Empty:
                    continue
            print("Stopped listening.")

    def start(self, duration=None, device=None):
        self.transcript = ""
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._recognition_worker, kwargs={"device": device}, daemon=True)
        self._thread.start()
        if duration:
            threading.Timer(duration, self.stop).start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.running = False

    def get_transcript(self):
        with self._lock:
            return self.transcript
