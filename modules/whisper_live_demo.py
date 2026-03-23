import threading
import queue
import sounddevice as sd
import numpy as np
import whisper

class WhisperRealtime:
    def __init__(self, model_name="large", language="vi", device=None, samplerate=16000, blocksize=4000, chunk_duration=6, chunk_overlap=1):
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
            print(f"[WhisperRealtime] Listening with adaptive Energy VAD (variable chunks, no overlap)...")
            
            # VAD params
            energy_threshold = 0.01  # Ngưỡng trung bình 0.01 để phát hiện có tiếng hay im lặng
            silence_duration = 1.2   # Tối thiểu 0.8s im lặng để chốt một câu
            max_duration = 20.0      # Ngắt câu sau 15 giây dù có đang nói (tránh đợi quá lâu)
            
            # State
            audio_buffer = np.array([], dtype=np.float32)
            silence_frames = 0
            is_speaking = False
            last_text = ""
            
            while not self._stop_event.is_set():
                try:
                    block = self.audio_queue.get(timeout=0.1)
                    
                    if block.ndim == 2 and block.shape[1] > 1:
                        block = block.mean(axis=1).astype(np.float32)
                    else:
                        block = block.flatten().astype(np.float32)
                        
                    # 1. Tính toán năng lượng âm thanh của block hiện tại để gom câu
                    energy = np.sqrt(np.mean(block**2))
                    
                    if energy > energy_threshold:
                        is_speaking = True
                        silence_frames = 0
                    elif is_speaking:
                        silence_frames += len(block)
                        
                    if is_speaking:
                        audio_buffer = np.concatenate((audio_buffer, block))
                        
                    current_duration = len(audio_buffer) / self.samplerate
                    
                    # 2. Quyết định ngắt câu khi đủ thời gian im lặng hoặc audio_buffer quá dài
                    if is_speaking and ((silence_frames / self.samplerate) >= silence_duration or current_duration >= max_duration):
                        if current_duration >= 0.5: # Chỉ nhận diện nếu có đoạn audio dài tối thiểu 0.5s
                            
                            # Cung cấp ngữ cảnh cho Whisper bằng đoạn chữ liền trước
                            prompt = (last_text + " ") if last_text else "Xin chào."
                            
                            result = self.model.transcribe(
                                audio_buffer, 
                                language=self.language, 
                                fp16=False, 
                                task="transcribe", 
                                condition_on_previous_text=False,
                                initial_prompt=prompt,
                                temperature=0.0,
                                beam_size=5,
                                no_speech_threshold=0.6,
                                logprob_threshold=-1.0,
                                compression_ratio_threshold=2.4
                            )
                            
                            # 3. Xóa ảo giác của model bằng cách quét qua từng phân đoạn (segment)
                            valid_segments_text = []
                            for seg in result.get("segments", []):
                                no_speech = seg.get("no_speech_prob", 0.0)
                                comp_ratio = seg.get("compression_ratio", 0.0)
                                
                                # Lọc nếu model không chắc có tiếng hoặc text bị nén (hay báo hiệu ảo giác)
                                if no_speech < 0.6 and comp_ratio < 2.0:
                                    valid_segments_text.append(seg["text"])
                            
                            text = "".join(valid_segments_text).strip()
                            
                            # 4. Lọc rác và ảo giác YouTube (Whisper hay tự bịa ra lúc im lặng hoặc nhạc nền)
                            text_lower = text.lower()
                            is_hallucination = False
                            
                            blocklist = [
                                "ghiền mì gõ", "subscribe", "đăng ký kênh", "đăng kí kênh", 
                                "theo dõi kênh", "nhấn chuông", "cảm ơn các bạn", "xin chào các bạn",
                                "chúc các bạn", "âm nhạc", "nhạc nền", "tiếng việt", "dấu chấm câu",
                                "amara.org", "bản quyền", "hẹn gặp lại", "vsub", "subtitles", "phụ đề"
                            ]
                            
                            for bad_phrase in blocklist:
                                if bad_phrase in text_lower:
                                    is_hallucination = True
                                    break
                                    
                            if is_hallucination:
                                text = ""  
                                
                            # 5. Filter mồi tĩnh quen thuộc nếu lỡ bị xuất
                            for hallucination in ["Xin chào.", "Chào các bạn.", "Đây là một cuộc họp", "Câu trước là:"]:
                                if hallucination in text:
                                    text = text.replace(hallucination, "").strip()
                            
                            # 6. Kiểm tra xem văn bản có bị lặp y hệt câu trước không (do mồi prompt)
                            if text and text == last_text:
                                text = ""
                                
                            if text:
                                print(f"[WhisperRealtime] Transcribed sentence ({current_duration:.1f}s): {text}")
                                from datetime import datetime
                                timestamp = datetime.now().strftime("[%H:%M]")
                                line = f"{timestamp} {text}"
                                with self._lock:
                                    self.transcript += ("\n" if self.transcript else "") + line
                                    
                                words = text.split()
                                last_text = " ".join(words[-15:]) if len(words) > 15 else text
                                
                        # 7. CHỐT CÂU: Xoá âm thanh đã xử lý, không có chồng chéo overlap
                        audio_buffer = np.array([], dtype=np.float32)
                        is_speaking = False
                        silence_frames = 0
                        
                except queue.Empty:
                    continue
            print("[WhisperRealtime] Stopped listening.")

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
