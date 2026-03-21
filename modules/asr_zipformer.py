"""Utilities for Vietnamese ASR inference using Zipformer models.

This module is built around sherpa-onnx and provides two main flows:
1) Offline/test inference from a wave file.
2) Real-time inference from microphone audio.

Expected model directory layout (example):
  model_dir/
    encoder-epoch-99-avg-1.int8.onnx
    decoder-epoch-99-avg-1.onnx
    joiner-epoch-99-avg-1.int8.onnx
    tokens.txt
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import sherpa_onnx
except ImportError as exc:  # pragma: no cover - import guard
    sherpa_onnx = None
    _SHERPA_IMPORT_ERROR = exc
else:
    _SHERPA_IMPORT_ERROR = None


@dataclass
class ZipformerModelPaths:
    """Holds resolved ONNX model file paths for Zipformer inference."""

    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path


def _require_sherpa_onnx() -> None:
    if sherpa_onnx is None:
        raise ImportError(
            "sherpa-onnx is required for ASR. Install with `pip install sherpa-onnx`."
        ) from _SHERPA_IMPORT_ERROR


def _first_match(model_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(model_dir.glob(pattern))
    return matches[0] if matches else None


def resolve_zipformer_model_paths(
    model_dir: str | Path,
    encoder_name: Optional[str] = None,
    decoder_name: Optional[str] = None,
    joiner_name: Optional[str] = None,
    tokens_name: str = "tokens.txt",
) -> ZipformerModelPaths:
    """Resolve model files from a Zipformer model directory.

    If explicit names are not provided, the function tries common filename patterns.
    """

    root = Path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"Model directory not found: {root}")

    encoder = root / encoder_name if encoder_name else _first_match(root, "encoder*.onnx")
    decoder = root / decoder_name if decoder_name else _first_match(root, "decoder*.onnx")
    joiner = root / joiner_name if joiner_name else _first_match(root, "joiner*.onnx")
    tokens = root / tokens_name

    missing = []
    if encoder is None or not encoder.exists():
        missing.append("encoder*.onnx")
    if decoder is None or not decoder.exists():
        missing.append("decoder*.onnx")
    if joiner is None or not joiner.exists():
        missing.append("joiner*.onnx")
    if not tokens.exists():
        missing.append(tokens_name)

    if missing:
        raise FileNotFoundError(
            f"Missing Zipformer files in {root}. Required: {', '.join(missing)}"
        )

    return ZipformerModelPaths(
        encoder=Path(encoder),
        decoder=Path(decoder),
        joiner=Path(joiner),
        tokens=tokens,
    )


class VietnameseZipformerASR:
    """Vietnamese ASR helper with offline and real-time inference methods."""

    def __init__(
        self,
        model_dir: str | Path,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        provider: str = "cpu",
        num_threads: int = 2,
        debug: bool = False,
        encoder_name: Optional[str] = None,
        decoder_name: Optional[str] = None,
        joiner_name: Optional[str] = None,
    ) -> None:
        _require_sherpa_onnx()

        self.sample_rate = sample_rate
        self.model_paths = resolve_zipformer_model_paths(
            model_dir=model_dir,
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            joiner_name=joiner_name,
        )

        self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer(
            encoder=str(self.model_paths.encoder),
            decoder=str(self.model_paths.decoder),
            joiner=str(self.model_paths.joiner),
            tokens=str(self.model_paths.tokens),
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            debug=debug,
            provider=provider,
        )

        self.online_recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer(
            encoder=str(self.model_paths.encoder),
            decoder=str(self.model_paths.decoder),
            joiner=str(self.model_paths.joiner),
            tokens=str(self.model_paths.tokens),
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            decoding_method="greedy_search",
            max_active_paths=4,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
            debug=debug,
            provider=provider,
        )

    def transcribe_wave_file(self, audio_path: str | Path) -> str:
        """Run offline inference from a wave file path."""
        _require_sherpa_onnx()

        stream = self.offline_recognizer.create_stream()
        samples = sherpa_onnx.read_wave(str(audio_path))
        stream.accept_waveform(self.sample_rate, samples)
        self.offline_recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str:
        """Run offline inference from a float waveform array in memory."""
        if samples.ndim != 1:
            raise ValueError("Audio array must be mono 1-D")

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample_rate={self.sample_rate}, got {sample_rate}. "
                "Please resample before calling transcribe_array()."
            )

        stream = self.offline_recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples.astype(np.float32))
        self.offline_recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def realtime_inference(
        self,
        duration_seconds: Optional[float] = None,
        on_partial_result: Optional[Callable[[str], None]] = None,
        on_final_result: Optional[Callable[[str], None]] = None,
        print_partial: bool = True,
    ) -> str:
        """Run real-time microphone inference.

        Args:
            duration_seconds: Stop automatically after this duration. None means keep running.
            on_partial_result: Optional callback for partial text updates.
            on_final_result: Optional callback each time endpoint is detected.
            print_partial: Whether to print partial text to stdout.

        Returns:
            The concatenated final transcript chunks.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "sounddevice is required for realtime inference. "
                "Install with `pip install sounddevice`."
            ) from exc

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        stop_event = threading.Event()
        stream = self.online_recognizer.create_stream()
        transcript_parts: list[str] = []
        last_partial = ""

        def _audio_callback(indata, frames, callback_time, status) -> None:
            del frames, callback_time
            if status:
                # Drop status logs to avoid spamming caller output.
                pass
            audio_queue.put(indata[:, 0].copy())

        def _loop() -> None:
            nonlocal last_partial
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                stream.accept_waveform(self.sample_rate, chunk)

                while self.online_recognizer.is_ready(stream):
                    self.online_recognizer.decode_stream(stream)

                partial = self.online_recognizer.get_result(stream).strip()
                if partial and partial != last_partial:
                    last_partial = partial
                    if print_partial:
                        print(f"\r[partial] {partial}", end="", flush=True)
                    if on_partial_result is not None:
                        on_partial_result(partial)

                if self.online_recognizer.is_endpoint(stream):
                    final_text = self.online_recognizer.get_result(stream).strip()
                    if final_text:
                        transcript_parts.append(final_text)
                        if print_partial:
                            print(f"\n[final] {final_text}")
                        if on_final_result is not None:
                            on_final_result(final_text)
                    self.online_recognizer.reset(stream)
                    last_partial = ""

        worker = threading.Thread(target=_loop, daemon=True)
        worker.start()

        start_time = time.time()
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            callback=_audio_callback,
        ):
            if duration_seconds is None:
                print("Realtime ASR is running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    pass
            else:
                while (time.time() - start_time) < duration_seconds:
                    time.sleep(0.2)

        stop_event.set()
        worker.join(timeout=2.0)

        # Flush final tail tokens after stream stops.
        while self.online_recognizer.is_ready(stream):
            self.online_recognizer.decode_stream(stream)
        tail = self.online_recognizer.get_result(stream).strip()
        if tail:
            transcript_parts.append(tail)

        return " ".join(part for part in transcript_parts if part).strip()
