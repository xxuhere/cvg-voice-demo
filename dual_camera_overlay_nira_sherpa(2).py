#!/usr/bin/env python3
"""
Dual Raspberry Pi Camera Module v2 overlay + direct MP4 recording + GPU display.

Camera 0: NIR/reference camera
Camera 1: visible camera, warped into the NIR coordinate system

Performance architecture
------------------------
* Main stream: selected capture resolution, YUV420, encoded directly by Picamera2.
* Recording: Picamera2 H264Encoder -> PyavOutput -> MP4.
* Lores stream: luminance (Y) copied only at the requested preview publication rate.
* Four-frame rings + SensorTimestamp matching choose the closest NIR/visible pair.
* GPU display: QOpenGLWidget + QPainter OpenGL paint engine perform:
    - visible->NIR projective transform
    - false-color composition
    - NIR-strength adjustment
  No cv2.remap(), no custom GL texture/shader wrapper, and no PyOpenGL.
* Mean2 (mean + 2 population SD) uses one preview histogram per displayed pair
  in a latest-pair worker. JET + threshold alpha use only a 256-entry LUT.
* Cached NIR Y statistics refresh at 5 Hz. Top10 is the upper-decile cutoff
  among above-threshold pixels, not their mean. Empty masks report N/A.
* Camera buffer_count is 16.
* Capture/recording FPS and overlay display FPS are independent.

Required packages
-----------------
sudo apt install -y python3-picamera2 python3-pyqt5 python3-av python3-opencv python3-scipy libportaudio2

Optional offline "CVG" voice control
-------------------------------------
1. In a virtual environment created with --system-site-packages, install:
       python -m pip install sherpa-onnx sounddevice sentencepiece scipy pywhispercpp
2. Download and extract the free English sherpa-onnx keyword model:
       sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
3. Download ggml-tiny.en.bin (recommended) and set WHISPER_MODEL_PATH.
   pywhispercpp loads this model once; whisper-cli is no longer required.
   Build whisper.cpp only if explicitly choosing CVG_WHISPER_BACKEND=cli.
4. Set these environment variables before starting the program:
       export SHERPA_KWS_MODEL_DIR='/path/to/sherpa-kws-model-folder'
       export WHISPER_CLI_PATH='/path/to/whisper.cpp/build/bin/whisper-cli'
       export WHISPER_MODEL_PATH='/path/to/ggml-tiny.en.bin'
       export CVG_AUDIO_DEVICE=4
       export CVG_KWS_THRESHOLD=0.20
       export CVG_KWS_SCORE=2.0

The older NIRA_* setting names remain accepted as fallbacks for compatibility.

The script creates the tokenized "CVG" keyword internally from bpe.model.
Sherpa-onnx listens continuously with one CPU inference thread; whisper.cpp
is replaced by persistent in-memory inference after the wake word is detected.
CVG_COMMAND_SECONDS defaults to 2.5; CVG_WHISPER_THREADS defaults to 2.
CVG_WHISPER_MAX_TOKENS defaults to 8 subword tokens, not eight words.
For longer future phrases, raise it to 12 or 16 if transcription is truncated.
Resolution (720p/1080p) and NIR/visible exposure phrases are registered but
currently return "GUI action not implemented yet" without changing settings.
Compare thread counts by restarting with CVG_WHISPER_THREADS=2 or 4 and
issuing identical commands while recording; inference logs include both
thread count and token limit. The script cannot benchmark Pi hardware itself.
CVG_WHISPER_AUDIO_CTX=auto reduces encoder context for short commands;
set it to 0 for full-context accuracy comparison. Greedy one-pass decoding
avoids expensive fallback retries; noisy speech may need repeating.
CVG_WHISPER_BACKEND=cli explicitly restores the original compatibility path. Voice processing runs in a separate
QThread, and all audio and recognition remain local on the Raspberry Pi.
Microphone rates are probed automatically: 16 kHz, 48 kHz, 44.1 kHz, then
the device default. Non-16-kHz audio is anti-alias filtered and resampled
to 16 kHz for wake words and commands (requires scipy).

Place homography.npy next to this script, or select it in the GUI.
The matrix must map visible-camera pixels to NIR-camera pixels.
"""

from __future__ import annotations

import csv
import math
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None  # NumPy fallback; OpenCV histogram avoids an int-sized frame copy.
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput
# Avoid the desktop-theme integration that triggers the GObject warning.
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"
os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import (
    QImage,
    QPainter,
    QTransform,
    QColor,
    qRgba,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QOpenGLWidget,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_HOMOGRAPHY = APP_DIR / "homography.npy"
DEFAULT_OUTPUT_DIR = Path.home() / "Videos" / "dual_camera"
DEFAULT_SHERPA_KWS_MODEL_DIR = (
    APP_DIR / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)
DEFAULT_WHISPER_MODEL = APP_DIR / "models" / "ggml-tiny.en.bin"

NIR_CAMERA_INDEX = 0
VISIBLE_CAMERA_INDEX = 1
HOMOGRAPHY_CALIBRATION_SIZE = (1280, 720)  # width, height
NOIR_TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"


VOICE_COMMANDS = {
    "start cameras": ("start cameras", "start camera", "turn on cameras", "camera start"),
    "stop cameras": ("stop cameras", "stop camera", "turn off cameras", "camera stop"),
    "start recording": ("start recording", "begin recording", "record both streams"),
    "stop recording": ("stop recording", "end recording", "finish recording"),
    "capture raw pair": (
        "capture raw pair", "capture raw", "capture bayer pair", "save raw pair",
        "take raw pair", "capture image",
    ),
    "increase overlay": (
        "increase overlay", "increase fluorescence", "more overlay", "overlay up",
    ),
    "decrease overlay": (
        "decrease overlay", "decrease fluorescence", "less overlay", "overlay down",
    ),
    # Future GUI controls: recognize complete phrases now, but route them to
    # an explicit pending-feature response until their actions are implemented.
    "change resolution to 720p": (
        "change resolution to 720p", "change resolution to 720",
        "set resolution to 720p", "set resolution to 720",
    ),
    "change resolution to 1080p": (
        "change resolution to 1080p", "change resolution to 1080",
        "set resolution to 1080p", "set resolution to 1080",
    ),
    "increase nir exposure": (
        "increase nir exposure", "increase n i r exposure",
        "increase near infrared exposure", "increase infrared exposure",
    ),
    "decrease nir exposure": (
        "decrease nir exposure", "decrease n i r exposure",
        "decrease near infrared exposure", "decrease infrared exposure",
    ),
    "increase visible camera exposure": (
        "increase visible camera exposure", "increase visible exposure",
    ),
    "decrease visible camera exposure": (
        "decrease visible camera exposure", "decrease visible exposure",
    ),
    "exit program": ("exit program", "close program", "quit program"),
    "confirm": ("confirm", "yes confirm"),
    "cancel": ("cancel", "never mind", "do not exit"),
}

PENDING_VOICE_COMMANDS = frozenset({
    "change resolution to 720p", "change resolution to 1080p",
    "increase nir exposure", "decrease nir exposure",
    "increase visible camera exposure", "decrease visible camera exposure",
})

def normalize_speech(text: str) -> str:
    """Normalize a Whisper transcript for conservative command matching."""
    text = text.lower()

    # Normalize likely transcriptions of the spoken letters “C V G”.
    text = re.sub(
        r"\b(?:c|see|sea)\s+(?:v|vee)\s+(?:g|gee)\b",
        "cvg",
        text,
    )
    text = text.replace("c v g", "cvg")
    text = text.replace("see vee gee", "cvg")
    text = text.replace("sea vee gee", "cvg")

    text = " ".join(re.findall(r"[a-z0-9]+", text))
    # Whisper may spell resolution numbers or separate the trailing P.
    text = re.sub(r"\b(?:seven twenty|seven hundred twenty|seven hundred and twenty)\b", "720", text)
    text = re.sub(r"\b(?:ten eighty|one thousand eighty|one thousand and eighty)\b", "1080", text)
    text = re.sub(r"\b(720|1080)\s+p\b", r"\1p", text)
    return text

def match_voice_command(text: str) -> Optional[str]:
    """Return a canonical command, preferring exact/contained phrases."""
    normalized = normalize_speech(text)
    if not normalized:
        return None
    candidates: list[tuple[float, str]] = []
    for command, aliases in VOICE_COMMANDS.items():
        for alias in aliases:
            if normalized == alias or f" {alias} " in f" {normalized} ":
                return command
            # Longer future phrases must be complete: a token-truncated
            # exposure command should not guess a camera or direction.
            if command in PENDING_VOICE_COMMANDS:
                continue
            score = difflib.SequenceMatcher(None, normalized, alias).ratio()
            candidates.append((score, command))
    if any(word in normalized.split() for word in ("exposure", "resolution", "nir", "visible")):
        return None
    score, command = max(candidates, default=(0.0, ""))
    return command if score >= 0.72 else None


def select_voice_capture_rate(sd, device) -> int:
    """Prefer model-native 16 kHz, then USB-friendly 48/44.1 kHz rates.

    PipeWire recording may work even when direct PortAudio/ALSA capture
    rejects 16 kHz. Probe the selected device rather than assuming support.
    """
    default_rate = int(round(sd.query_devices(device, "input")["default_samplerate"]))
    failures = []
    for rate in dict.fromkeys((16000, 48000, 44100, default_rate)):
        # Exact 100 ms blocks must map to exactly 1600 recognition samples.
        if rate <= 0 or rate % 10:
            continue
        try:
            sd.check_input_settings(device=device, channels=1,
                                    dtype="float32", samplerate=rate)
            return rate
        except Exception as exc:
            failures.append(f"{rate} Hz: {exc}")
    raise RuntimeError("No supported mono capture rate for the selected microphone. "
                       + "; ".join(failures))


def resample_voice_audio(audio: np.ndarray, capture_rate: int) -> np.ndarray:
    """Anti-aliased conversion to 16 kHz for Sherpa and whisper.cpp.

    48 kHz uses a 1:3 ratio; 44.1 kHz uses 160:441. Merely taking every
    third sample would alias high-frequency microphone noise into speech.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if capture_rate == 16000:
        return audio
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Audio resampling requires scipy. Install python3-scipy "
                           "or run python -m pip install scipy in your environment.") from exc
    divisor = math.gcd(capture_rate, 16000)
    return resample_poly(audio, 16000 // divisor,
                         capture_rate // divisor).astype(np.float32, copy=False)


class VoiceBlockResampler:
    """Resample continuous 100 ms wake-word blocks with filter context.

    Keep one block of history and one block of lookahead so independently
    resampled blocks do not acquire zero-padding artifacts at every join.
    This adds 100 ms of buffering only when rate conversion is needed.
    Reset after overflow or command capture because those break continuity.
    """
    def __init__(self, capture_rate: int):
        self.capture_rate = capture_rate
        self.history = None
        self.pending = None

    def push(self, audio: np.ndarray) -> np.ndarray:
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.capture_rate == 16000:
            return block
        if block.size != self.capture_rate // 10:
            raise ValueError("Expected one 100 ms microphone block")
        if self.pending is None:
            self.pending = block.copy()
            return np.empty(0, dtype=np.float32)
        left = self.history if self.history is not None else np.zeros_like(self.pending)
        converted = resample_voice_audio(np.concatenate((left, self.pending, block)),
                                        self.capture_rate)
        self.history, self.pending = self.pending, block.copy()
        return converted[1600:3200]


class BufferedVoiceInput:
    """PortAudio callback drains capture independently of recognition speed.

    The callback only copies audio into a bounded mailbox (no inference,
    filtering or Qt signals). On backlog, discard stale blocks together so
    listening resumes on current audio instead of accumulating latency.
    """
    def __init__(self, sd, stop_event, **settings):
        self.condition = threading.Condition()
        self.blocks = deque()
        self.discontinuous = False
        self.device_overflows = 0
        self.backlog_drops = 0
        self.stop_event = stop_event
        self.stream = sd.InputStream(callback=self._capture, latency="high", **settings)

    def _capture(self, audio, frames, timing, status):
        with self.condition:
            # PortAudio flags an upstream loss, but the queued samples are
            # still valid. Never erase them on each flag: doing so repeatedly
            # starves the wake-word decoder. Count hardware flags separately.
            if status.input_overflow:
                self.device_overflows += 1
            if len(self.blocks) >= 10:
                self.blocks.clear()
                self.discontinuous = True
                self.backlog_drops += 1
            self.blocks.append(audio.copy())
            self.condition.notify()

    def read(self, count):
        with self.condition:
            while not self.blocks:
                if self.stop_event.is_set():
                    return np.zeros((count, 1), np.float32), False
                if not self.stream.active:
                    raise RuntimeError("Microphone capture stopped unexpectedly")
                self.condition.wait(timeout=0.2)
            audio = self.blocks.popleft()
            gap, self.discontinuous = self.discontinuous, False
        if audio.shape[0] != count:
            raise RuntimeError("Unexpected microphone callback block length")
        return audio, gap

    def diagnostics(self):
        with self.condition:
            return self.device_overflows, self.backlog_drops, len(self.blocks)

    def discard_pending(self):
        """Start a command on fresh audio, clearing wake-word backlog flags."""
        with self.condition:
            self.blocks.clear()
            self.discontinuous = False

    def stop(self):
        self.stream.stop()
        with self.condition:
            self.blocks.clear()
            self.discontinuous = False

    def start(self):
        with self.condition:
            self.blocks.clear()
            self.discontinuous = False
        self.stream.start()

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


class VoiceWorker(QThread):
    """Sherpa-onnx "CVG" spotting + whisper.cpp command recognition."""

    status_changed = pyqtSignal(str)
    command_detected = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        kws_encoder: Path,
        kws_decoder: Path,
        kws_joiner: Path,
        kws_tokens: Path,
        kws_bpe_model: Path,
        whisper_cli: Path,
        whisper_model: Path,
        command_seconds: float = 2.5,
        audio_device_index: int = -1,
        keywords_score: float = 1.5,
        keywords_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.kws_encoder = kws_encoder
        self.kws_decoder = kws_decoder
        self.kws_joiner = kws_joiner
        self.kws_tokens = kws_tokens
        self.kws_bpe_model = kws_bpe_model
        self.whisper_cli = whisper_cli
        self.whisper_model = whisper_model
        self.command_seconds = command_seconds
        self.audio_device_index = audio_device_index
        self.keywords_score = keywords_score
        self.keywords_threshold = keywords_threshold
        self._stop_event = threading.Event()
        self._process_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._persistent_whisper = None
        self._last_transcription_seconds = 0.0

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def _load_fast_whisper(self) -> None:
        """Load once before microphone capture; never reload per command.

        CLI mode is an explicit compatibility option, not a silent slow fallback.
        Two inference threads leave resources for dual cameras on the Pi 5.
        """
        backend = os.environ.get("CVG_WHISPER_BACKEND", "persistent").lower()
        if backend == "cli":
            print("VOICE: CLI compatibility mode (model reloads per command)", flush=True)
            return
        if backend != "persistent":
            raise ValueError("CVG_WHISPER_BACKEND must be persistent or cli")
        try:
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise RuntimeError("Fast voice requires pywhispercpp: run python -m pip install "
                               "--upgrade pywhispercpp in the CVG environment. "
                               "For old CLI mode set CVG_WHISPER_BACKEND=cli.") from exc
        threads = int(os.environ.get("CVG_WHISPER_THREADS", "2"))
        if not 1 <= threads <= 4:
            raise ValueError("CVG_WHISPER_THREADS must be 1 through 4")
        # Tokens are subword units, not words. Eight reduces repetitive output;
        # raise to 12 or 16 if future longer phrases are cut off. Read once at
        # voice startup so comparisons use a fixed configuration per session.
        max_tokens = int(os.environ.get("CVG_WHISPER_MAX_TOKENS", "8"))
        if not 1 <= max_tokens <= 64:
            raise ValueError("CVG_WHISPER_MAX_TOKENS must be 1 through 64")
        self._whisper_threads = threads
        self._whisper_max_tokens = max_tokens
        started = time.perf_counter()
        self.status_changed.emit("VOICE: loading persistent Whisper model…")
        self._persistent_whisper = Model(
            str(self.whisper_model), params_sampling_strategy=0,
            n_threads=threads, language="en", no_context=True,
            single_segment=True, no_timestamps=True,
            print_progress=False, print_realtime=False, print_timestamps=False,
            temperature=0.0, temperature_inc=0.0, greedy={"best_of": 1},
            max_tokens=max_tokens,
            initial_prompt=", ".join(VOICE_COMMANDS),
        )
        print(f"VOICE TIMING: model load {time.perf_counter()-started:.3f}s; "
              f"threads={threads}; max_tokens={max_tokens}; model={self.whisper_model.name}", flush=True)

    def _transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._persistent_whisper is None:
            return self._transcribe_cli(samples, sample_rate)
        if sample_rate != 16000:
            raise ValueError("Whisper input must be genuinely resampled to 16 kHz")
        # Pass normalized audio directly; no int16 WAV or child process needed.
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.dtype == np.int16:
            audio /= 32768.0
        duration = audio.size / sample_rate
        setting = os.environ.get("CVG_WHISPER_AUDIO_CTX", "auto").lower()
        # Whisper normally uses the full encoder context. Short commands can
        # use a smaller context (50 encoder positions/second plus padding).
        # This is a speed/accuracy tradeoff: set AUDIO_CTX=0 for full context.
        context = (min(1500, max(100, int(math.ceil((duration + 1.0) * 50))))
                   if setting == "auto" else int(setting))
        if not 0 <= context <= 1500:
            raise ValueError("CVG_WHISPER_AUDIO_CTX must be auto or 0 through 1500")
        if context and context < math.ceil(duration * 50):
            raise ValueError("Whisper audio context is too short for the recorded command")
        started = time.perf_counter()
        # Current bindings support cooperative cancellation; older bindings
        # finish the active decode before stopping. Do not forcibly kill QThread.
        import inspect
        kwargs = {"audio_ctx": context}
        if "abort_callback" in inspect.signature(self._persistent_whisper.transcribe).parameters:
            kwargs["abort_callback"] = self._stop_event.is_set
        segments = self._persistent_whisper.transcribe(audio, **kwargs)
        self._last_transcription_seconds = time.perf_counter() - started
        print(f"VOICE TIMING: inference {self._last_transcription_seconds:.3f}s; "
              f"audio={duration:.2f}s; audio_ctx={context}; "
              f"threads={self._whisper_threads}; max_tokens={self._whisper_max_tokens}", flush=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _transcribe_cli(self, samples: np.ndarray, sample_rate: int) -> str:
        with tempfile.TemporaryDirectory(prefix="cvg_voice_") as temp_dir:
            temp = Path(temp_dir)
            wav_path = temp / "command.wav"
            output_base = temp / "transcript"
            pcm = np.asarray(samples, dtype=np.int16)
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())

            prompt = ", ".join(VOICE_COMMANDS.keys())
            args = [
                str(self.whisper_cli), "-m", str(self.whisper_model),
                "-f", str(wav_path), "-l", "en", "-nt", "-np",
                "-otxt", "-of", str(output_base), "--prompt", prompt,
            ]
            process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            with self._process_lock:
                self._process = process
            try:
                while process.poll() is None:
                    if self._stop_event.wait(0.05):
                        process.terminate()
                        raise RuntimeError("voice control stopped")
                stdout, stderr = process.communicate()
            finally:
                with self._process_lock:
                    self._process = None

            if process.returncode != 0:
                detail = (stderr or stdout or "unknown whisper.cpp error").strip()
                raise RuntimeError(f"whisper.cpp failed: {detail[-500:]}")
            transcript_path = output_base.with_suffix(".txt")
            if transcript_path.exists():
                return transcript_path.read_text(encoding="utf-8", errors="replace").strip()
            return stdout.strip()


    def _create_keyword_file(self, directory: Path) -> Path:
        """Create the Sherpa-ONNX wake-word file for CVG."""
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "Voice package missing: install sentencepiece in the active environment."
            ) from exc

        processor = spm.SentencePieceProcessor(model_file=str(self.kws_bpe_model))

        # Spaces encourage recognition of the three spoken letters:
        # “C” “V” “G”.
        pieces = processor.encode("C V G", out_type=str)

        if not pieces:
            raise RuntimeError("The Sherpa keyword model could not tokenize CVG")

        valid_tokens = set()
        with self.kws_tokens.open("r", encoding="utf-8") as token_file:
            for line in token_file:
                fields = line.rstrip("\n").rsplit(maxsplit=1)
                if fields:
                    valid_tokens.add(fields[0])
        missing = [piece for piece in pieces if piece not in valid_tokens]
        if missing:
            raise RuntimeError(
                "Sherpa BPE/token files do not match; missing tokens: "
                + ", ".join(missing)
            )

        keyword_path = directory / "cvg_keywords.txt"
        keyword_path.write_text(
            " ".join(pieces)
            + f" :{self.keywords_score:g} #{self.keywords_threshold:g} @CVG\n",
            encoding="utf-8",
        )
        return keyword_path
    
    def run(self) -> None:
        try:
            try:
                import sherpa_onnx
                import sounddevice as sd
            except ImportError as exc:
                raise RuntimeError(
                    "Voice packages missing. Install sherpa-onnx, sounddevice, "
                    "and sentencepiece in the active environment."
                ) from exc

            # Capture and recognition rates are deliberately separate.
            # USB PnP microphones often reject 16 kHz but accept 48/44.1 kHz.
            self._load_fast_whisper()
            if self._stop_event.is_set():
                return
            sample_rate = 16000  # Recognition/WAV rate; never relabel native audio.
            input_device = None if self.audio_device_index < 0 else self.audio_device_index
            capture_rate = select_voice_capture_rate(sd, input_device)
            samples_per_read = capture_rate // 10  # 100 ms at the actual capture rate.
            # Validate the optional dependency before opening the microphone.
            resample_voice_audio(np.zeros(samples_per_read, np.float32), capture_rate)
            wake_resampler = VoiceBlockResampler(capture_rate)

            with tempfile.TemporaryDirectory(prefix="cvg_kws_") as temp_dir:
                keyword_path = self._create_keyword_file(Path(temp_dir))
                keyword_spotter = sherpa_onnx.KeywordSpotter(
                    tokens=str(self.kws_tokens),
                    encoder=str(self.kws_encoder),
                    decoder=str(self.kws_decoder),
                    joiner=str(self.kws_joiner),
                    keywords_file=str(keyword_path),
                    num_threads=1,
                    max_active_paths=4,
                    keywords_score=self.keywords_score,
                    keywords_threshold=self.keywords_threshold,
                    num_trailing_blanks=1,
                    provider="cpu",
                )
                kws_stream = keyword_spotter.create_stream()

                with BufferedVoiceInput(
                    sd, self._stop_event,
                    device=input_device,
                    channels=1,
                    dtype="float32",
                    samplerate=capture_rate,
                    blocksize=samples_per_read,
                ) as audio_stream:
                    ready_message = (f"VOICE: READY — input {capture_rate} Hz → "
                                     "16000 Hz; say ‘C V G’")
                    self.status_changed.emit(ready_message)
                    last_overflow_notice = 0.0
                    next_audio_diagnostic = time.monotonic() + 5.0
                    print(ready_message, flush=True)

                    while not self._stop_event.is_set():
                        audio, overflowed = audio_stream.read(samples_per_read)
                        now = time.monotonic()
                        if now >= next_audio_diagnostic:
                            hardware, backlog, queued = audio_stream.diagnostics()
                            print(f"VOICE AUDIO: device-overflow flags={hardware}; "
                                  f"backlog drops={backlog}; queued blocks={queued}", flush=True)
                            next_audio_diagnostic = now + 5.0
                        if overflowed:
                            # Only a real mailbox discard resets decoder history.
                            # Hardware flags alone are counted, not repeatedly
                            # used to erase the accumulated wake-word context.
                            # A gap invalidates decoder/filter history. Report
                            # recovery sparingly, then restore READY explicitly.
                            now = time.monotonic()
                            if now - last_overflow_notice >= 5.0:
                                self.status_changed.emit("VOICE: recovering microphone backlog")
                                last_overflow_notice = now
                            self.status_changed.emit(
                                "VOICE: READY — recognition backlog discarded; say C V G"
                            )
                            wake_resampler = VoiceBlockResampler(capture_rate)
                            kws_stream = keyword_spotter.create_stream()
                        wake_audio = wake_resampler.push(audio)
                        if wake_audio.size == 0:
                            continue  # First converted block is waiting for lookahead.
                        kws_stream.accept_waveform(sample_rate, wake_audio)

                        detected = False
                        while keyword_spotter.is_ready(kws_stream):
                            keyword_spotter.decode_stream(kws_stream)
                            if keyword_spotter.get_result(kws_stream):
                                keyword_spotter.reset_stream(kws_stream)
                                detected = True
                                break
                        if not detected:
                            continue

                        # Keyword inference can leave older blocks queued.
                        # Clear them BEFORE notifying the user to speak, so
                        # recording begins with fresh command audio and does
                        # not inherit an overflow flag from wake-word capture.
                        audio_stream.discard_pending()
                        hardware_at_command = audio_stream.diagnostics()[0]
                        print("VOICE: wake word detected; capturing command", flush=True)
                        self.status_changed.emit("VOICE: LISTENING…")
                        command_blocks = max(
                            1,
                            int(np.ceil(
                                self.command_seconds * capture_rate / samples_per_read
                            )),
                        )
                        capture_started = time.perf_counter()
                        blocks = []
                        command_gap_count = 0
                        for _ in range(command_blocks):
                            if self._stop_event.is_set():
                                break
                            block, gap = audio_stream.read(samples_per_read)
                            command_gap_count += int(gap)
                            blocks.append(block.reshape(-1).copy())
                        if self._stop_event.is_set():
                            break

                        # PortAudio overflow flags mean some input was lost,
                        # but do not make every remaining sample unusable. Let
                        # Whisper transcribe what was captured instead of rejecting
                        # every command on a single flag. Keep the warning visible;
                        # a badly damaged phrase can still fail command matching.
                        device_flags = audio_stream.diagnostics()[0] - hardware_at_command
                        capture_note = (
                            f" | backlog interruptions {command_gap_count}, device flags {device_flags}"
                            if command_gap_count or device_flags else ""
                        )
                        if command_gap_count:
                            print(f"VOICE: command capture reported {command_gap_count} "
                                  "interruptions; transcribing available audio", flush=True)

                        capture_finished = time.perf_counter()
                        # Resample the entire command once, preserving filter
                        # continuity across its captured blocks, then create a
                        # genuine 16 kHz WAV for whisper.cpp.
                        command_audio = np.concatenate(blocks)[
                            : int(round(self.command_seconds * capture_rate))]
                        command_audio = resample_voice_audio(command_audio, capture_rate)
                        command_audio = command_audio[
                            : int(round(self.command_seconds * sample_rate))]
                        resample_finished = time.perf_counter()
                        pcm16 = np.asarray(
                            np.clip(command_audio, -1.0, 1.0) * 32767.0,
                            dtype=np.int16,
                        )

                        # Stop/flush PortAudio while whisper.cpp runs so old audio
                        # is not mistaken for a new wake word when listening resumes.
                        audio_stream.stop()
                        try:
                            self.status_changed.emit("VOICE: RECOGNIZING…" + capture_note)
                            # In persistent mode retain float audio precision.
                            decode_audio = command_audio if self._persistent_whisper is not None else pcm16
                            transcript = self._transcribe(decode_audio, sample_rate)
                            print(f"VOICE TIMING: capture={capture_finished-capture_started:.3f}s; "
                                  f"resample={resample_finished-capture_finished:.3f}s; "
                                  f"post-capture={time.perf_counter()-capture_finished:.3f}s", flush=True)
                        finally:
                            if not self._stop_event.is_set():
                                audio_stream.start()
                                kws_stream = keyword_spotter.create_stream()
                                wake_resampler = VoiceBlockResampler(capture_rate)

                        command = match_voice_command(transcript)
                        # Preserve the result while listening for the next wake
                        # word. Previously READY immediately hid both rejected
                        # transcripts and GUI command feedback.
                        print(f"VOICE transcript: {transcript!r}; matched: {command!r}{capture_note}", flush=True)
                        if command is None:
                            rms = float(np.sqrt(np.mean(command_audio ** 2)))
                            self.status_changed.emit(
                                f"VOICE: READY — NOT UNDERSTOOD: {transcript or '(silence)'} "
                                f"| audio RMS {rms:.4f}{capture_note}; say C V G to retry"
                            )
                        else:
                            # The GUI command handler supplies its own feedback;
                            # do not overwrite it with a generic READY message.
                            print(f"VOICE TIMING: dispatch command={command!r} "
                                  f"monotonic={time.perf_counter():.6f}", flush=True)
                            self.command_detected.emit(command, transcript)

        except Exception as exc:
            if not self._stop_event.is_set():
                self.error.emit(str(exc))


@dataclass(frozen=True)
class FramePacket:
    y: np.ndarray
    sensor_timestamp_ns: int
    sequence: int
    monotonic_ns: int


class FrameRingBuffer:
    """Small thread-safe ring buffer used only by the live preview path.

    Recording does not pass through this buffer.  A depth of four provides a
    short history so the display scheduler can choose the NIR/visible pair
    with the smallest SensorTimestamp difference instead of simply pairing the
    two most recent frames.
    """

    def __init__(self, maxlen: int = 4) -> None:
        self._lock = threading.Lock()
        self._frames = deque(maxlen=maxlen)

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self._frames.append(packet)

    def snapshot(self) -> list[FramePacket]:
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class CameraWorker(QThread):
    error = pyqtSignal(str)
    started_ok = pyqtSignal(int)

    def __init__(
        self,
        camera_index: int,
        size: Tuple[int, int],
        fps: int,
        frame_buffer: FrameRingBuffer,
        preview_size: Tuple[int, int],
        preview_publish_fps: int,
        noir_tuning: bool = False,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.size = size
        self.preview_size = preview_size
        self.fps = fps
        self.frame_buffer = frame_buffer
        self.preview_publish_fps = max(1, int(preview_publish_fps))
        self.noir_tuning = noir_tuning
        self._stop_event = threading.Event()
        self._camera: Optional[Picamera2] = None

        self._record_lock = threading.Lock()
        self._record_command: Optional[tuple[str, Optional[Path], int]] = None
        self._record_command_event = threading.Event()
        self._record_done_event = threading.Event()
        self._record_error: Optional[str] = None
        self._recording = False

        self._raw_lock = threading.Lock()
        self._raw_command: Optional[tuple[Path, Path]] = None
        self._raw_command_event = threading.Event()
        self._raw_done_event = threading.Event()
        self._raw_error: Optional[str] = None
        self._encoder: Optional[H264Encoder] = None
        self._output: Optional[PyavOutput] = None
        self._record_start_time = 0.0
        self._timestamp_rows: list[tuple[int, int, int]] = []
        self._timestamp_path: Optional[Path] = None

    def request_stop(self) -> None:
        self._stop_event.set()

    def _submit_record_command(
        self,
        command: str,
        path: Optional[Path] = None,
        bitrate_bps: int = 10_000_000,
        timeout: float = 12.0,
    ) -> None:
        with self._record_lock:
            self._record_error = None
            self._record_done_event.clear()
            self._record_command = (command, path, bitrate_bps)
            self._record_command_event.set()
        if not self._record_done_event.wait(timeout=timeout):
            raise RuntimeError(
                f"Camera {self.camera_index}: timed out while trying to {command} recording"
            )
        with self._record_lock:
            error = self._record_error
        if error:
            raise RuntimeError(error)

    def start_direct_recording(self, path: Path, bitrate_bps: int) -> None:
        self._submit_record_command("start", path, bitrate_bps)

    def stop_direct_recording(self) -> None:
        if self.is_recording():
            self._submit_record_command("stop")

    def capture_raw_pair_member(self, dng_path: Path, npy_path: Path, timeout: float = 12.0) -> None:
        with self._raw_lock:
            self._raw_error = None
            self._raw_done_event.clear()
            self._raw_command = (dng_path, npy_path)
            self._raw_command_event.set()
        if not self._raw_done_event.wait(timeout=timeout):
            raise RuntimeError(
                f"Camera {self.camera_index}: timed out while trying to capture raw Bayer image"
            )
        with self._raw_lock:
            error = self._raw_error
        if error:
            raise RuntimeError(error)

    def is_recording(self) -> bool:
        with self._record_lock:
            return self._recording

    def recording_stats(self) -> tuple[bool, int, float]:
        with self._record_lock:
            recording = self._recording
            encoder = self._encoder
            start_time = self._record_start_time
        frames = int(getattr(encoder, "frames_encoded", 0)) if encoder is not None else 0
        elapsed = time.monotonic() - start_time if recording and start_time > 0 else 0.0
        fps = frames / elapsed if elapsed > 0.25 else 0.0
        return recording, frames, fps

    def _write_timestamp_csv(self) -> None:
        path = self._timestamp_path
        rows = self._timestamp_rows
        if path is None or not rows:
            return
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["capture_sequence", "sensor_timestamp_ns", "host_monotonic_ns"])
                writer.writerows(rows)
        except Exception as exc:
            self.error.emit(f"Camera {self.camera_index}: timestamp CSV: {exc}")

    def _handle_raw_command(self, request=None) -> None:
        if not self._raw_command_event.is_set():
            return
        with self._raw_lock:
            command = self._raw_command
            self._raw_command = None
            self._raw_command_event.clear()
        if command is None:
            return

        dng_path, npy_path = command
        error: Optional[str] = None
        try:
            if request is None:
                raise RuntimeError("raw Bayer capture requires an active camera request")
            dng_path.parent.mkdir(parents=True, exist_ok=True)
            raw_array = np.array(request.make_array("raw"), copy=True, order="C")
            request.save_dng(str(dng_path))
            np.save(npy_path, raw_array)
        except Exception as exc:
            error = f"Camera {self.camera_index}: raw Bayer capture failed: {exc}"
        finally:
            with self._raw_lock:
                self._raw_error = error
            self._raw_done_event.set()

    def _handle_record_command(self) -> None:
        if not self._record_command_event.is_set():
            return
        with self._record_lock:
            command = self._record_command
            self._record_command = None
            self._record_command_event.clear()
        if command is None:
            return

        action, path, bitrate_bps = command
        error: Optional[str] = None
        try:
            if self._camera is None:
                raise RuntimeError("camera is not initialized")

            if action == "start":
                if self._recording:
                    raise RuntimeError("recording is already active")
                if path is None:
                    raise RuntimeError("recording path was not supplied")
                path.parent.mkdir(parents=True, exist_ok=True)
                encoder = H264Encoder(
                    bitrate=int(bitrate_bps),
                    framerate=self.fps,
                    iperiod=max(1, self.fps),
                    repeat=True,
                )
                output = PyavOutput(str(path))
                self._camera.start_encoder(encoder, output, name="main")

                self._timestamp_path = path.with_name(
                    f"{path.stem}_capture_timestamps.csv"
                )
                self._timestamp_rows = []
                with self._record_lock:
                    self._encoder = encoder
                    self._output = output
                    self._record_start_time = time.monotonic()
                    self._recording = True

            elif action == "stop":
                with self._record_lock:
                    encoder = self._encoder
                if encoder is not None:
                    self._camera.stop_encoder(encoder)
                with self._record_lock:
                    self._recording = False
                    self._encoder = None
                    self._output = None
                self._write_timestamp_csv()
                self._timestamp_rows = []
                self._timestamp_path = None
            else:
                raise RuntimeError(f"unknown recording command: {action}")

        except Exception as exc:
            error = f"Camera {self.camera_index}: {action} recording failed: {exc}"
            with self._record_lock:
                self._recording = False
                self._encoder = None
                self._output = None
        finally:
            with self._record_lock:
                self._record_error = error
            self._record_done_event.set()

    def run(self) -> None:
        try:
            kwargs = {}
            if self.noir_tuning and Path(NOIR_TUNING_FILE).exists():
                kwargs["tuning"] = Picamera2.load_tuning_file(NOIR_TUNING_FILE)

            self._camera = Picamera2(self.camera_index, **kwargs)
            frame_duration_us = int(round(1_000_000 / self.fps))
            use_lores = self.preview_size != self.size
            config_args = dict(
                main={"size": self.size, "format": "YUV420"},
                raw={},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                buffer_count=16,
                queue=False,
            )
            if use_lores:
                config_args["lores"] = {"size": self.preview_size, "format": "YUV420"}

            config = self._camera.create_video_configuration(**config_args)
            self._camera.configure(config)
            self._camera.start()
            self.started_ok.emit(self.camera_index)

            sequence = 0
            pw, ph = self.preview_size

            # Fractional publication accumulator.  For example, at 30 FPS
            # capture and 24 FPS display it publishes four of every five
            # camera frames instead of copying all 30 lores images into Python.
            # At 24/24 every frame is published.
            publish_accumulator = float(self.fps)
            publish_rate = min(self.preview_publish_fps, self.fps)

            while not self._stop_event.is_set():
                self._handle_record_command()
                request = self._camera.capture_request()
                y_plane = None
                try:
                    metadata = request.get_metadata()
                    sequence += 1
                    sensor_ts = int(metadata.get("SensorTimestamp", time.monotonic_ns()))
                    host_ts = time.monotonic_ns()

                    self._handle_raw_command(request)

                    publish_accumulator += publish_rate
                    publish_preview = publish_accumulator >= self.fps
                    if publish_preview:
                        publish_accumulator -= self.fps
                        display_name = "lores" if use_lores else "main"
                        display_array = request.make_array(display_name)
                        # Only the Y plane is copied, and only for frames selected
                        # for live preview publication.  Direct H.264 recording
                        # remains independent and receives every main-stream frame.
                        y_plane = np.array(
                            display_array[:ph, :pw], dtype=np.uint8, copy=True, order="C"
                        )
                finally:
                    request.release()

                if y_plane is not None:
                    self.frame_buffer.put(
                        FramePacket(y_plane, sensor_ts, sequence, host_ts)
                    )

                if self.is_recording():
                    self._timestamp_rows.append((sequence, sensor_ts, host_ts))

            if self.is_recording() and self._camera is not None:
                try:
                    with self._record_lock:
                        encoder = self._encoder
                    if encoder is not None:
                        self._camera.stop_encoder(encoder)
                finally:
                    with self._record_lock:
                        self._recording = False
                        self._encoder = None
                        self._output = None
                    self._write_timestamp_csv()

        except Exception as exc:
            self.error.emit(f"Camera {self.camera_index}: {exc}")
        finally:
            if self._raw_command_event.is_set():
                with self._raw_lock:
                    self._raw_error = f"Camera {self.camera_index}: camera stopped before raw Bayer capture completed"
                    self._raw_command = None
                    self._raw_command_event.clear()
                self._raw_done_event.set()
            self._record_done_event.set()
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception:
                    pass
                try:
                    self._camera.close()
                except Exception:
                    pass
                self._camera = None


class DisplayScheduler(QThread):
    """Timestamp-match preview frames at the requested display cadence.

    The old scheduler required *both* cameras to advance their sequence number
    before emitting another display frame.  Two free-running cameras can be
    almost one frame apart, so that rule unnecessarily reduced display FPS.

    This scheduler keeps a short history from each camera and, at every display
    tick, chooses the available pair with the smallest absolute
    SensorTimestamp difference.  At least one member of the pair must be new;
    therefore one camera's previous frame may be reused for a single display
    cycle when that gives a much better temporal match.
    """

    frame_pair = pyqtSignal(object, object, float, float)
    error = pyqtSignal(str)

    def __init__(
        self,
        nir_buffer: FrameRingBuffer,
        visible_buffer: FrameRingBuffer,
        display_fps: int,
    ) -> None:
        super().__init__()
        self.nir_buffer = nir_buffer
        self.visible_buffer = visible_buffer
        self.display_fps = max(1, int(display_fps))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _best_pair(
        nir_frames: list[FramePacket],
        vis_frames: list[FramePacket],
        last_nir_seq: int,
        last_vis_seq: int,
        max_age_ns: int,
    ) -> Optional[tuple[FramePacket, FramePacket]]:
        if not nir_frames or not vis_frames:
            return None

        newest_ts = max(
            nir_frames[-1].sensor_timestamp_ns,
            vis_frames[-1].sensor_timestamp_ns,
        )
        cutoff = newest_ts - max_age_ns
        nir_candidates = [p for p in nir_frames if p.sensor_timestamp_ns >= cutoff]
        vis_candidates = [p for p in vis_frames if p.sensor_timestamp_ns >= cutoff]

        candidates = []
        for nir in nir_candidates:
            for vis in vis_candidates:
                # Do not emit the identical pair twice, but permit one side to
                # be reused if the other camera has advanced.
                if nir.sequence == last_nir_seq and vis.sequence == last_vis_seq:
                    continue
                delta = abs(vis.sensor_timestamp_ns - nir.sensor_timestamp_ns)
                # For equal timestamp error, prefer the more recent pair.
                recency = max(nir.sensor_timestamp_ns, vis.sensor_timestamp_ns)
                candidates.append((delta, -recency, nir, vis))

        if not candidates:
            return None
        _, _, nir, vis = min(candidates, key=lambda item: (item[0], item[1]))
        return nir, vis

    def run(self) -> None:
        try:
            target_period = 1.0 / self.display_fps
            next_deadline = time.perf_counter()
            last_nir_seq = -1
            last_vis_seq = -1
            count = 0
            fps_start = next_deadline
            measured_fps = 0.0
            # Enough history to match a camera that is nearly one complete
            # frame out of phase, without intentionally adding large latency.
            max_age_ns = int(2.5 * 1_000_000_000 / self.display_fps)

            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(min(next_deadline - now, 0.002))
                    continue
                next_deadline += target_period
                if next_deadline < now - target_period:
                    next_deadline = now + target_period

                nir_frames = self.nir_buffer.snapshot()
                vis_frames = self.visible_buffer.snapshot()
                pair = self._best_pair(
                    nir_frames, vis_frames, last_nir_seq, last_vis_seq, max_age_ns
                )
                if pair is None:
                    continue

                nir, vis = pair
                last_nir_seq = nir.sequence
                last_vis_seq = vis.sequence

                count += 1
                now = time.perf_counter()
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    measured_fps = count / elapsed
                    count = 0
                    fps_start = now

                pair_delta_us = (
                    vis.sensor_timestamp_ns - nir.sensor_timestamp_ns
                ) / 1000.0
                self.frame_pair.emit(
                    nir.y, vis.y, measured_fps, pair_delta_us
                )
        except Exception as exc:
            self.error.emit(f"Display scheduler: {exc}")


def fluorescence_histogram(nir: np.ndarray, method: str = "Mean2") -> dict:
    """One uint8 preview histogram; no sorting or full-frame float conversion.

    Top10 is the histogram cutoff for the brightest 10% of above-threshold
    pixels, following the repository's cumulative-histogram definition.
    Intensities are processed Y-plane digital numbers, not raw Bayer values.
    """
    if nir.dtype != np.uint8 or nir.ndim != 2 or nir.size == 0:
        raise ValueError("Fluorescence preview must be a nonempty uint8 Y-plane")
    started = time.perf_counter()
    hist = (cv2.calcHist([nir], [0], None, [256], [0, 256]).ravel().astype(np.float64)
            if cv2 is not None else np.bincount(nir.ravel(), minlength=256).astype(np.float64))
    levels = np.arange(256, dtype=np.float64)
    count = float(hist.sum())
    mean = float(hist @ levels / count)
    sigma = float(np.sqrt(max(0.0, hist @ (levels * levels) / count - mean * mean)))
    threshold = mean + 2.0 * sigma if method == "Mean2" else -1.0
    cutoff = int(np.floor(threshold))
    cumulative = hist.cumsum()
    below = float(cumulative[min(cutoff, 255)]) if cutoff >= 0 else 0.0
    above = count - below
    top10 = int(np.searchsorted(cumulative, count - 0.1 * above)) if above else None
    high = int(np.flatnonzero(hist)[-1])
    return dict(mean=mean, sigma=sigma, threshold=threshold, cutoff=cutoff,
                top10=top10, high=high, above_percent=100.0 * above / count,
                analysis_ms=1000.0 * (time.perf_counter() - started))


class FluorescenceAnalyzer:
    """Bounded latest-pair mailbox; computation never runs in camera callbacks.

    Both input and output hold at most one pair. GUI polling avoids a backlog
    of queued Qt signals; each result carries its matching images and settings.
    """
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="CVG-fluorescence")
        self.thread.start()

    def submit(self, nir, visible, method, generation):
        with self.condition:
            self.pending = (nir, visible, method, generation)
            self.condition.notify()

    def take(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                nir, visible, method, generation = self.pending
                self.pending = None
            try:
                stats = fluorescence_histogram(nir, method)
                result = (nir, visible, method, generation, stats, None)
            except Exception as exc:
                result = (nir, visible, method, generation, None, str(exc))
            with self.condition:
                if not self.stopped:
                    self.result = result

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = self.result = None
            self.condition.notify()
        self.thread.join(timeout=1.0)


class GLOverlayWidget(QOpenGLWidget):
    """Crash-resistant GPU-assisted overlay using QPainter on QOpenGLWidget.

    This version deliberately avoids QOpenGLTexture, PyOpenGL, raw GL texture
    calls, and custom shader objects.  Those wrappers can trigger native
    segmentation faults on some Raspberry Pi OS / PyQt5 / Mesa combinations.

    Qt documents QPainter-on-QOpenGLWidget as a hardware-accelerated 2-D path
    when supported by the OpenGL driver.  The visible image is projectively
    transformed with QTransform; the NIR image is used as the reference and as
    a thresholded JET false-colour overlay.
    """

    gl_error = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self._nir: Optional[np.ndarray] = None
        self._visible: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._homography = np.eye(3, dtype=np.float64)
        self._qtransform = QTransform()
        self._strength = 0.70
        self._method = "Mean2"
        self._show_statistics = True
        self._statistics_text = ""
        self._text_updated = 0.0
        self._stats = None
        self._lut_key = None
        self._generation = 0
        self._analyzer = None
        if cv2 is not None:
            self._jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                                         cv2.COLORMAP_JET).reshape(256, 3)[:, ::-1]
        else:
            # Standard JET ramp; OpenCV installation gives its exact palette.
            t = np.arange(256, dtype=np.float64) / 255.0
            self._jet = np.rint(255 * np.clip(np.stack(
                [1.5 - np.abs(4*t - 3), 1.5 - np.abs(4*t - 2),
                 1.5 - np.abs(4*t - 1)], axis=1), 0, 1)).astype(np.uint8)
        self._result_timer = QTimer(self)
        self._result_timer.setInterval(5)
        self._result_timer.timeout.connect(self._accept_analysis)
        self._nir_lut = []
        self._rebuild_nir_lut()
        self.rendered_frames = 0
        self.render_fps = 0.0
        self._fps_count = 0
        self._fps_start = time.perf_counter()
        self.setAutoFillBackground(False)

    def _rebuild_nir_lut(self) -> None:
        alpha = int(round(255.0 * self._strength))
        stats = self._stats
        cutoff = stats["cutoff"] if stats else 255
        high = stats["high"] if stats else 255
        top10 = stats["top10"] if stats else None
        key = (self._method, alpha, cutoff, high, top10)
        if key == self._lut_key:
            return
        self._lut_key = key
        if self._method == "None":
            cutoff, high, top10 = 0, 255, None
        table = []
        for i in range(256):
            if i <= cutoff or high <= cutoff:
                table.append(qRgba(0, 0, 0, 0))
                continue
            # Repository scaling: threshold->observed maximum, with the
            # brightest 10% saturated at the upper end of JET.
            level = 255 if top10 is not None and i > top10 else int(
                np.clip(255.0 * (i - cutoff) / (high - cutoff), 0, 255))
            r, g, b = map(int, self._jet[level])
            table.append(qRgba(r, g, b, alpha))
        self._nir_lut = table

    def set_threshold_method(self, method: str) -> None:
        if method not in ("Mean2", "None"):
            return
        self._method = method
        self._generation += 1
        self._statistics_text = ""
        if self._nir is not None:
            self.set_frames(self._nir, self._visible)

    def set_statistics_visible(self, visible: bool) -> None:
        self._show_statistics = bool(visible)
        self.update()

    def _accept_analysis(self) -> None:
        if self._analyzer is None:
            return
        result = self._analyzer.take()
        if result is None:
            return
        nir, visible, method, generation, stats, error = result
        if generation != self._generation:
            return
        if error:
            self.gl_error.emit("Fluorescence analysis failed: " + error)
            return
        self._nir, self._visible, self._stats = nir, visible, stats
        self._frame_size = (nir.shape[1], nir.shape[0])
        self._rebuild_nir_lut()
        now = time.perf_counter()
        if not self._statistics_text or now - self._text_updated >= 0.2:
            top = str(stats["top10"]) if stats["top10"] is not None else "N/A"
            threshold = f'{stats["threshold"]:.2f}' if method == "Mean2" else "OFF"
            self._statistics_text = (
                f'NIR Y | {method} | Mean {stats["mean"]:.2f} | SD {stats["sigma"]:.2f}\n'
                f'Threshold {threshold} | Top10 cutoff {top}\n'
                f'Above threshold {stats["above_percent"]:.2f}% | Analysis {stats["analysis_ms"]:.2f} ms')
            self._text_updated = now
        self.update()

    def set_homography(self, homography: np.ndarray, size: Tuple[int, int]) -> None:
        self._homography = np.asarray(homography, dtype=np.float64).copy()
        h = self._homography
        # QTransform's matrix layout differs from the conventional homography
        # array.  This maps source visible pixel coordinates into NIR pixels.
        self._qtransform = QTransform(
            float(h[0, 0]), float(h[1, 0]), float(h[2, 0]),
            float(h[0, 1]), float(h[1, 1]), float(h[2, 1]),
            float(h[0, 2]), float(h[1, 2]), float(h[2, 2]),
        )
        self.update()

    def set_nir_strength(self, strength: float) -> None:
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._rebuild_nir_lut()
        self.update()

    @pyqtSlot(object, object)
    def set_frames(self, nir: np.ndarray, visible: np.ndarray) -> None:
        if nir.ndim != 2 or visible.ndim != 2 or nir.shape != visible.shape:
            self.gl_error.emit(
                f"Unexpected preview shapes: NIR {nir.shape}, visible {visible.shape}"
            )
            return
        if nir.dtype != np.uint8 or visible.dtype != np.uint8:
            self.gl_error.emit("Preview Y-planes must have uint8 dtype")
            return
        if self._analyzer is None:
            self._analyzer = FluorescenceAnalyzer()
            self._result_timer.start()
        self._analyzer.submit(nir, visible, self._method, self._generation)

    @staticmethod
    def _gray_qimage(array: np.ndarray) -> QImage:
        a = np.ascontiguousarray(array)
        h, w = a.shape
        # The NumPy array remains referenced by the widget while paintGL runs,
        # so QImage may safely wrap it without an additional full-frame copy.
        return QImage(a.data, w, h, int(a.strides[0]), QImage.Format_Grayscale8)

    def _pseudo_qimage(self, array: np.ndarray) -> QImage:
        a = np.ascontiguousarray(array)
        h, w = a.shape
        image = QImage(a.data, w, h, int(a.strides[0]), QImage.Format_Indexed8)
        image.setColorTable(self._nir_lut)
        return image

    def paintGL(self) -> None:
        if self._nir is None or self._visible is None or self._frame_size is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            painter.end()
            return

        try:
            nir = self._nir
            visible = self._visible
            src_w, src_h = self._frame_size

            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.fillRect(self.rect(), Qt.black)

            # Fit the reference image to the widget while preserving aspect ratio.
            widget_w = max(1, self.width())
            widget_h = max(1, self.height())
            scale = min(widget_w / src_w, widget_h / src_h)
            draw_w = src_w * scale
            draw_h = src_h * scale
            off_x = (widget_w - draw_w) * 0.5
            off_y = (widget_h - draw_h) * 0.5

            # Use logical NIR pixels as painter coordinates. The viewport maps
            # those logical pixels to the fitted widget rectangle.
            painter.setWindow(0, 0, src_w, src_h)
            painter.setViewport(
                int(round(off_x)), int(round(off_y)),
                max(1, int(round(draw_w))), max(1, int(round(draw_h)))
            )

            nir_gray = self._gray_qimage(nir)
            visible_gray = self._gray_qimage(visible)
            nir_color = self._pseudo_qimage(nir)

            # 1) NIR grayscale is the reference/background.
            painter.setWorldTransform(QTransform(), False)
            painter.setOpacity(1.0)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.drawImage(0, 0, nir_gray)

            # 2) Warp the visible luminance into NIR coordinates. Qt's OpenGL
            # paint engine performs the projective texture mapping when supported.
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setWorldTransform(self._qtransform, False)
            painter.drawImage(0, 0, visible_gray)

            # 3) Add the NIR signal as a thresholded JET overlay in the NIR
            # coordinate system. Alpha is encoded in the indexed-colour LUT.
            painter.setWorldTransform(QTransform(), False)
            painter.setCompositionMode(QPainter.CompositionMode_Plus)
            painter.drawImage(0, 0, nir_color)
            if self._show_statistics and self._statistics_text:
                # Screen coordinates keep text readable at all preview sizes.
                painter.setWindow(self.rect())
                painter.setViewport(self.rect())
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                font = painter.font()
                font.setPixelSize(13)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                lines = self._statistics_text.split("\n")
                x, y = int(round(off_x)) + 8, int(round(off_y)) + 8
                width = max(metrics.horizontalAdvance(line) for line in lines) + 16
                height = metrics.height() * len(lines) + 12
                painter.fillRect(x, y, width, height, QColor(0, 0, 0, 180))
                painter.setPen(Qt.white)
                for index, line in enumerate(lines):
                    painter.drawText(x + 8, y + 6 + metrics.ascent() + index * metrics.height(), line)
            painter.end()

            self.rendered_frames += 1
            self._fps_count += 1
            now = time.perf_counter()
            elapsed = now - self._fps_start
            if elapsed >= 1.0:
                self.render_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_start = now

        except Exception as exc:
            try:
                if 'painter' in locals() and painter.isActive():
                    painter.end()
            except Exception:
                pass
            self.gl_error.emit(f"OpenGL/QPainter render failed: {exc}")

    def cleanup(self) -> None:
        self._generation += 1
        self._result_timer.stop()
        if self._analyzer is not None:
            self._analyzer.close()
            self._analyzer = None
        self._stats = None
        self._statistics_text = ""
        self._lut_key = None
        # No manually-managed native GL resources in this implementation.
        self._nir = None
        self._visible = None

def load_and_scale_homography(path: Path, runtime_size: Tuple[int, int]) -> np.ndarray:
    h = np.load(path).astype(np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("Homography must be a finite 3x3 NumPy array")
    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Homography is singular or nearly singular")

    cal_w, cal_h = HOMOGRAPHY_CALIBRATION_SIZE
    run_w, run_h = runtime_size
    s = np.array(
        [[run_w / cal_w, 0, 0], [0, run_h / cal_h, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    runtime_h = s @ h @ np.linalg.inv(s)
    if abs(runtime_h[2, 2]) > 1e-12:
        runtime_h /= runtime_h[2, 2]
    return runtime_h


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dual Camera Overlay - Timestamp Matched QPainter Display")
        self.nir_buffer = FrameRingBuffer(maxlen=4)
        self.visible_buffer = FrameRingBuffer(maxlen=4)
        self.nir_worker: Optional[CameraWorker] = None
        self.visible_worker: Optional[CameraWorker] = None
        self.display_scheduler: Optional[DisplayScheduler] = None
        self.voice_worker: Optional[VoiceWorker] = None
        self.recording = False
        self.exit_confirmation_deadline = 0.0
        self.homography_path = DEFAULT_HOMOGRAPHY
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.last_scheduler_fps = 0.0
        self.last_delta_us = 0.0
        self._build_ui()
        self._update_controls(False)

    def _build_ui(self) -> None:
        self.display = GLOverlayWidget(self)
        self.display.gl_error.connect(self._show_error)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080", "1280x720", "640x480"])
        self.resolution_combo.setCurrentText("1920x1080")

        self.preview_resolution_combo = QComboBox()
        self.preview_resolution_combo.addItems([
            "Auto (960x540 for 1080p)",
            "1280x720",
            "960x540",
            "Native capture resolution",
        ])
        self.preview_resolution_combo.setCurrentText("1280x720")

        self.capture_fps_combo = QComboBox()
        self.capture_fps_combo.addItems(["15", "24", "30"])
        self.capture_fps_combo.setCurrentText("30")

        self.display_fps_combo = QComboBox()
        self.display_fps_combo.addItems(["20", "24", "30"])
        self.display_fps_combo.setCurrentText("24")

        self.nir_slider = QSlider(Qt.Horizontal)
        self.nir_slider.setRange(0, 100)
        self.nir_slider.setValue(70)

        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(2, 30)
        self.bitrate_spin.setValue(10)
        self.bitrate_spin.setSuffix(" Mb/s per camera")

        self.noir_check = QCheckBox("Use IMX219 NoIR tuning for camera 0")
        self.noir_check.setChecked(True)

        self.start_button = QPushButton("Start Cameras")
        self.stop_button = QPushButton("Stop Cameras")
        self.record_button = QPushButton("Start Recording Both Streams")
        self.stop_record_button = QPushButton("Stop Recording")
        self.snapshot_button = QPushButton("Capture Raw Bayer Pair")
        self.homography_button = QPushButton("Select Homography")
        self.output_button = QPushButton("Select Output Folder")
        self.voice_button = QPushButton("Enable CVG Voice")
        self.voice_button.setCheckable(True)
        self.status = QLabel("Idle")
        self.status.setWordWrap(True)
        self.voice_status = QLabel("VOICE: OFF")
        self.voice_status.setWordWrap(True)
        self.voice_alert = QLabel("● STANDBY")
        self.voice_alert.setAlignment(Qt.AlignCenter)
        self.voice_alert.setMinimumWidth(135)
        self.voice_alert_flash_on = False
        self.voice_alert_timer = QTimer(self)
        self.voice_alert_timer.setInterval(300)
        self.voice_alert_timer.timeout.connect(self._toggle_voice_alert)
        self._set_voice_alert(False)

        controls = QGridLayout()
        controls.addWidget(QLabel("Capture resolution"), 0, 0)
        controls.addWidget(self.resolution_combo, 0, 1)
        controls.addWidget(QLabel("Capture/record FPS"), 0, 2)
        controls.addWidget(self.capture_fps_combo, 0, 3)
        controls.addWidget(QLabel("GPU overlay resolution"), 1, 0)
        controls.addWidget(self.preview_resolution_combo, 1, 1)
        controls.addWidget(QLabel("Display target FPS"), 1, 2)
        controls.addWidget(self.display_fps_combo, 1, 3)
        controls.addWidget(QLabel("NIR overlay strength"), 2, 0)
        controls.addWidget(self.nir_slider, 2, 1)
        controls.addWidget(QLabel("Recording bitrate"), 2, 2)
        controls.addWidget(self.bitrate_spin, 2, 3)
        controls.addWidget(self.noir_check, 3, 0, 1, 4)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["Mean2", "None"])
        self.threshold_combo.currentTextChanged.connect(self.display.set_threshold_method)
        self.statistics_check = QCheckBox("Show fluorescence statistics (5 Hz)")
        self.statistics_check.setChecked(True)
        self.statistics_check.toggled.connect(self.display.set_statistics_visible)
        controls.addWidget(QLabel("Fluorescence threshold"), 4, 0)
        controls.addWidget(self.threshold_combo, 4, 1)
        controls.addWidget(self.statistics_check, 4, 2, 1, 2)

        row1 = QHBoxLayout()
        row1.addWidget(self.start_button)
        row1.addWidget(self.stop_button)
        row1.addWidget(self.snapshot_button)
        row2 = QHBoxLayout()
        row2.addWidget(self.record_button)
        row2.addWidget(self.stop_record_button)
        row2.addWidget(self.homography_button)
        row2.addWidget(self.output_button)
        row2.addWidget(self.voice_button)

        root = QVBoxLayout()
        root.addWidget(self.display, 1)
        root.addLayout(controls)
        root.addLayout(row1)
        root.addLayout(row2)
        root.addWidget(self.status)
        voice_status_row = QHBoxLayout()
        voice_status_row.addWidget(self.voice_alert)
        voice_status_row.addWidget(self.voice_status, 1)
        root.addLayout(voice_status_row)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.start_button.clicked.connect(self.start_cameras)
        self.stop_button.clicked.connect(self.stop_cameras)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.snapshot_button.clicked.connect(self.capture_raw_bayer_pair)
        self.homography_button.clicked.connect(self.select_homography)
        self.output_button.clicked.connect(self.select_output_dir)
        self.voice_button.toggled.connect(self.toggle_voice_control)
        self.nir_slider.valueChanged.connect(
            lambda value: self.display.set_nir_strength(value / 100.0)
        )

    def _size(self) -> Tuple[int, int]:
        return tuple(map(int, self.resolution_combo.currentText().split("x")))  # type: ignore

    def _preview_size(self) -> Tuple[int, int]:
        cw, ch = self._size()
        mode = self.preview_resolution_combo.currentText()
        if mode.startswith("Auto"):
            return (960, 540) if cw > 1280 or ch > 720 else (cw, ch)
        if mode == "1280x720":
            return (min(1280, cw), min(720, ch))
        if mode == "960x540":
            return (min(960, cw), min(540, ch))
        return (cw, ch)

    def _update_controls(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.record_button.setEnabled(running and not self.recording)
        self.stop_record_button.setEnabled(self.recording)
        self.snapshot_button.setEnabled(running)
        for widget in (
            self.resolution_combo,
            self.preview_resolution_combo,
            self.capture_fps_combo,
            self.display_fps_combo,
            self.noir_check,
            self.homography_button,
        ):
            widget.setEnabled(not running)

    def _show_error(self, message: str) -> None:
        self.status.setText(message)
        QMessageBox.critical(self, "Error", message)

    def _apply_voice_alert_style(self) -> None:
        if not self.voice_alert_timer.isActive():
            self.voice_alert.setStyleSheet(
                "QLabel { color: #b0b0b0; background-color: #303030; "
                "border: 2px solid #606060; border-radius: 8px; "
                "font-weight: bold; padding: 5px; }"
            )
        elif self.voice_alert_flash_on:
            self.voice_alert.setStyleSheet(
                "QLabel { color: #08230f; background-color: #52ff7f; "
                "border: 2px solid #c8ffd5; border-radius: 8px; "
                "font-weight: bold; padding: 5px; }"
            )
        else:
            self.voice_alert.setStyleSheet(
                "QLabel { color: #d2ffdc; background-color: #12652d; "
                "border: 2px solid #35b95c; border-radius: 8px; "
                "font-weight: bold; padding: 5px; }"
            )

    def _set_voice_alert(self, listening: bool) -> None:
        if listening:
            self.voice_alert.setText("● LISTENING")
            self.voice_alert_flash_on = True
            if not self.voice_alert_timer.isActive():
                self.voice_alert_timer.start()
        else:
            self.voice_alert_timer.stop()
            self.voice_alert_flash_on = False
            self.voice_alert.setText("● STANDBY")
        self._apply_voice_alert_style()

    def _toggle_voice_alert(self) -> None:
        self.voice_alert_flash_on = not self.voice_alert_flash_on
        self._apply_voice_alert_style()

    @pyqtSlot(str)
    def _update_voice_status(self, message: str) -> None:
        self.voice_status.setText(message)
        self._set_voice_alert(message.startswith("VOICE: LISTENING"))

    def _voice_error(self, message: str) -> None:
        self._set_voice_alert(False)
        self.voice_status.setText(f"VOICE ERROR: {message}")
        self.voice_button.blockSignals(True)
        self.voice_button.setChecked(False)
        self.voice_button.setText("Enable CVG Voice")
        self.voice_button.blockSignals(False)
        worker = self.voice_worker
        if worker is not None and worker.isRunning():
            worker.request_stop()

    def _voice_worker_finished(self) -> None:
        self._set_voice_alert(False)
        worker = self.voice_worker
        if worker is not None:
            worker.deleteLater()
        self.voice_worker = None
        if self.voice_button.isChecked():
            self.voice_button.blockSignals(True)
            self.voice_button.setChecked(False)
            self.voice_button.setText("Enable CVG Voice")
            self.voice_button.blockSignals(False)

    def toggle_voice_control(self, enabled: bool) -> None:
        if not enabled:
            self.stop_voice_control()
            return

        kws_model_dir = Path(
            os.environ.get("SHERPA_KWS_MODEL_DIR", str(DEFAULT_SHERPA_KWS_MODEL_DIR))
        ).expanduser()
        whisper_model = Path(
            os.environ.get("WHISPER_MODEL_PATH", str(DEFAULT_WHISPER_MODEL))
        ).expanduser()
        configured_cli = os.environ.get("WHISPER_CLI_PATH", "").strip()
        discovered_cli = configured_cli or shutil.which("whisper-cli") or ""
        whisper_cli = Path(discovered_cli).expanduser() if discovered_cli else Path()

        def select_kws_model(component: str) -> Path:
            configured = os.environ.get(
                f"SHERPA_KWS_{component.upper()}", ""
            ).strip()
            if configured:
                return Path(configured).expanduser()
            candidates = list(kws_model_dir.glob(f"{component}-*.onnx"))
            candidates.sort(
                key=lambda path: (
                    ".int8.onnx" not in path.name,
                    "chunk-16" not in path.name,
                    path.name,
                )
            )
            return candidates[0] if candidates else kws_model_dir / f"{component}.onnx"

        kws_encoder = select_kws_model("encoder")
        kws_decoder = select_kws_model("decoder")
        kws_joiner = select_kws_model("joiner")
        kws_tokens = Path(
            os.environ.get("SHERPA_KWS_TOKENS", str(kws_model_dir / "tokens.txt"))
        ).expanduser()
        kws_bpe_model = Path(
            os.environ.get("SHERPA_KWS_BPE_MODEL", str(kws_model_dir / "bpe.model"))
        ).expanduser()

        problems = []
        if not kws_model_dir.is_dir():
            problems.append(f"Sherpa KWS model folder not found: {kws_model_dir}")
        for label, path in (
            ("encoder", kws_encoder),
            ("decoder", kws_decoder),
            ("joiner", kws_joiner),
            ("tokens.txt", kws_tokens),
            ("bpe.model", kws_bpe_model),
        ):
            if not path.is_file():
                problems.append(f"Sherpa {label} not found: {path}")
        if os.environ.get("CVG_WHISPER_BACKEND", "persistent").lower() == "cli" and (not discovered_cli or not whisper_cli.is_file()):
            problems.append("whisper-cli not found; set WHISPER_CLI_PATH")
        if not whisper_model.is_file():
            problems.append(f"Whisper model not found: {whisper_model}")
        if problems:
            self.voice_button.blockSignals(True)
            self.voice_button.setChecked(False)
            self.voice_button.blockSignals(False)
            self.voice_status.setText("VOICE: SETUP REQUIRED — " + "; ".join(problems))
            return

        try:
            command_seconds = float(
                os.environ.get(
                    "CVG_COMMAND_SECONDS",
                    os.environ.get("NIRA_COMMAND_SECONDS", "2.5"),
                )
            )
            audio_device_index = int(
                os.environ.get(
                    "CVG_AUDIO_DEVICE",
                    os.environ.get("NIRA_AUDIO_DEVICE", "-1"),
                )
            )
            keywords_score = float(
                os.environ.get(
                    "CVG_KWS_SCORE",
                    os.environ.get("NIRA_KWS_SCORE", "1.5"),
                )
            )
            keywords_threshold = float(
                os.environ.get(
                    "CVG_KWS_THRESHOLD",
                    os.environ.get("NIRA_KWS_THRESHOLD", "0.35"),
                )
            )
        except ValueError:
            self.voice_button.blockSignals(True)
            self.voice_button.setChecked(False)
            self.voice_button.blockSignals(False)
            self.voice_status.setText(
                "VOICE: invalid CVG command, device, score, or threshold setting"
            )
            return
        if command_seconds <= 0 or keywords_score <= 0 or not 0 < keywords_threshold < 1:
            self.voice_button.blockSignals(True)
            self.voice_button.setChecked(False)
            self.voice_button.blockSignals(False)
            self.voice_status.setText(
                "VOICE: seconds/score must be >0 and threshold must be between 0 and 1"
            )
            return

        self.voice_worker = VoiceWorker(
            kws_encoder=kws_encoder,
            kws_decoder=kws_decoder,
            kws_joiner=kws_joiner,
            kws_tokens=kws_tokens,
            kws_bpe_model=kws_bpe_model,
            whisper_cli=whisper_cli,
            whisper_model=whisper_model,
            command_seconds=command_seconds,
            audio_device_index=audio_device_index,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
        )
        self.voice_worker.status_changed.connect(self._update_voice_status)
        self.voice_worker.command_detected.connect(self.handle_voice_command)
        self.voice_worker.error.connect(self._voice_error)
        self.voice_worker.finished.connect(self._voice_worker_finished)
        self.voice_button.setText("Disable CVG Voice")
        self._set_voice_alert(False)
        self.voice_status.setText("VOICE: STARTING…")
        self.voice_worker.start()

    def stop_voice_control(self) -> bool:
        self._set_voice_alert(False)
        worker = self.voice_worker
        if worker is not None:
            worker.request_stop()
            if not worker.wait(5000):
                self.voice_status.setText("VOICE: waiting for recognition process to stop…")
                return False
        self.voice_button.setText("Enable CVG Voice")
        self.voice_status.setText("VOICE: OFF")
        return True

    @pyqtSlot(str, str)
    def handle_voice_command(self, command: str, transcript: str) -> None:
        print(f"VOICE TIMING: GUI received command={command!r} "
              f"monotonic={time.perf_counter():.6f}", flush=True)
        heard = transcript.strip() or command
        self._set_voice_alert(False)
        self.voice_status.setText(f"VOICE HEARD: “{heard}” → {command}")

        if command in PENDING_VOICE_COMMANDS:
            # Integration point for forthcoming resolution/exposure GUI actions.
            # Do not silently change camera settings or restart recording now.
            self.voice_status.setText(
                f"VOICE: recognized '{command}' — GUI action not implemented yet"
            )
            return
        if command == "start cameras":
            if self.nir_worker is None:
                self.start_cameras()
            else:
                self.voice_status.setText("VOICE: cameras are already running")
        elif command == "stop cameras":
            if self.nir_worker is not None:
                self.stop_cameras()
            else:
                self.voice_status.setText("VOICE: cameras are already stopped")
        elif command == "start recording":
            if self.nir_worker is None:
                self.voice_status.setText("VOICE: start cameras before recording")
            elif not self.recording:
                self.start_recording()
            else:
                self.voice_status.setText("VOICE: recording is already active")
        elif command == "stop recording":
            if self.recording:
                self.stop_recording()
            else:
                self.voice_status.setText("VOICE: recording is already stopped")
        elif command == "capture raw pair":
            if self.nir_worker is not None:
                self.capture_raw_bayer_pair()
            else:
                self.voice_status.setText("VOICE: start cameras before raw capture")
        elif command == "increase overlay":
            self.nir_slider.setValue(min(100, self.nir_slider.value() + 10))
            self.voice_status.setText(
                f"VOICE: NIR overlay increased to {self.nir_slider.value()}%"
            )
        elif command == "decrease overlay":
            self.nir_slider.setValue(max(0, self.nir_slider.value() - 10))
            self.voice_status.setText(
                f"VOICE: NIR overlay decreased to {self.nir_slider.value()}%"
            )
        elif command == "exit program":
            self.exit_confirmation_deadline = time.monotonic() + 10.0
            self.voice_status.setText(
                "VOICE: say ‘C V G’, then say ‘confirm’ within 10 seconds to exit"
            )
        elif command == "confirm":
            if time.monotonic() <= self.exit_confirmation_deadline:
                self.exit_confirmation_deadline = 0.0
                self.close()
            else:
                self.voice_status.setText("VOICE: nothing is awaiting confirmation")
        elif command == "cancel":
            self.exit_confirmation_deadline = 0.0
            self.voice_status.setText("VOICE: exit cancelled")

    def select_homography(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select homography.npy", str(self.homography_path.parent), "NumPy files (*.npy)"
        )
        if filename:
            self.homography_path = Path(filename)

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select output folder", str(self.output_dir)
        )
        if directory:
            self.output_dir = Path(directory)

    def start_cameras(self) -> None:
        if not self.homography_path.exists():
            self._show_error(f"Homography file not found: {self.homography_path}")
            return
        try:
            size = self._size()
            preview_size = self._preview_size()
            capture_fps = int(self.capture_fps_combo.currentText())
            display_fps = int(self.display_fps_combo.currentText())
            homography = load_and_scale_homography(self.homography_path, preview_size)
            self.display.set_homography(homography, preview_size)
            self.display.set_nir_strength(self.nir_slider.value() / 100.0)

            self.nir_buffer.clear()
            self.visible_buffer.clear()
            self.nir_worker = CameraWorker(
                NIR_CAMERA_INDEX, size, capture_fps, self.nir_buffer,
                preview_size, display_fps, self.noir_check.isChecked(),
            )
            self.visible_worker = CameraWorker(
                VISIBLE_CAMERA_INDEX, size, capture_fps, self.visible_buffer,
                preview_size, display_fps, False,
            )
            self.display_scheduler = DisplayScheduler(
                self.nir_buffer, self.visible_buffer, display_fps
            )
            for worker in (self.nir_worker, self.visible_worker):
                worker.error.connect(self._show_error)
            self.display_scheduler.error.connect(self._show_error)
            self.display_scheduler.frame_pair.connect(self._update_gpu_frames)

            self.nir_worker.start()
            self.visible_worker.start()
            self.display_scheduler.start()
            self.status.setText(
                f"Running: record {size[0]}x{size[1]}@{capture_fps}; "
                f"GPU overlay {preview_size[0]}x{preview_size[1]} target {display_fps} FPS"
            )
            self._update_controls(True)
        except Exception as exc:
            self._show_error(str(exc))
            self.stop_cameras()

    @pyqtSlot(object, object, float, float)
    def _update_gpu_frames(
        self, nir: np.ndarray, visible: np.ndarray, scheduler_fps: float, delta_us: float
    ) -> None:
        self.last_scheduler_fps = scheduler_fps
        self.last_delta_us = delta_us
        self.display.set_frames(nir, visible)
        if self.recording and self.nir_worker is not None and self.visible_worker is not None:
            _, f0, e0 = self.nir_worker.recording_stats()
            _, f1, e1 = self.visible_worker.recording_stats()
            self.status.setText(
                f"Direct recording | scheduler {scheduler_fps:.1f} FPS | "
                f"GL painter {self.display.render_fps:.1f} FPS | pair delta {delta_us:+.0f} us | "
                f"encoder C0/C1 {e0:.1f}/{e1:.1f} FPS | frames {f0}/{f1}"
            )
        else:
            self.status.setText(
                f"GPU painter display | scheduler {scheduler_fps:.1f} FPS | "
                f"GL painter {self.display.render_fps:.1f} FPS | pair delta {delta_us:+.0f} us"
            )

    def stop_cameras(self) -> None:
        self.stop_recording()
        if self.display_scheduler is not None:
            self.display_scheduler.request_stop()
            self.display_scheduler.wait(3000)
            self.display_scheduler = None
        for name in ("nir_worker", "visible_worker"):
            worker = getattr(self, name)
            if worker is not None:
                worker.request_stop()
                worker.wait(5000)
                setattr(self, name, None)
        self.display.cleanup()
        self.display.update()
        self.status.setText("Cameras stopped")
        self._update_controls(False)

    def start_recording(self) -> None:
        if self.nir_worker is None or self.visible_worker is None or self.recording:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bitrate = self.bitrate_spin.value() * 1_000_000
            paths = [
                self.output_dir / f"{stamp}_camera0_NIR.mp4",
                self.output_dir / f"{stamp}_camera1_visible.mp4",
            ]
            errors: list[str] = []

            def start_one(worker: CameraWorker, path: Path) -> None:
                try:
                    worker.start_direct_recording(path, bitrate)
                except Exception as exc:
                    errors.append(str(exc))

            threads = [
                threading.Thread(target=start_one, args=(self.nir_worker, paths[0])),
                threading.Thread(target=start_one, args=(self.visible_worker, paths[1])),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise RuntimeError("; ".join(errors))
            self.recording = True
            self._update_controls(True)
        except Exception as exc:
            self.recording = False
            self._show_error(str(exc))

    def stop_recording(self) -> None:
        if not self.recording:
            return
        errors: list[str] = []
        workers = [w for w in (self.nir_worker, self.visible_worker) if w is not None]

        def stop_one(worker: CameraWorker) -> None:
            try:
                worker.stop_direct_recording()
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=stop_one, args=(w,)) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.recording = False
        self.status.setText(
            "Recording stopped" if not errors else "Recording stopped with errors: " + "; ".join(errors)
        )
        self._update_controls(self.nir_worker is not None)

    def capture_raw_bayer_pair(self) -> None:
        if self.nir_worker is None or self.visible_worker is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dng_paths = [
                self.output_dir / f"{stamp}_camera0_NIR_raw.dng",
                self.output_dir / f"{stamp}_camera1_visible_raw.dng",
            ]
            npy_paths = [
                self.output_dir / f"{stamp}_camera0_NIR_raw.npy",
                self.output_dir / f"{stamp}_camera1_visible_raw.npy",
            ]
            errors: list[str] = []

            def capture_one(worker: CameraWorker, dng_path: Path, npy_path: Path) -> None:
                try:
                    worker.capture_raw_pair_member(dng_path, npy_path)
                except Exception as exc:
                    errors.append(str(exc))

            threads = [
                threading.Thread(target=capture_one, args=(self.nir_worker, dng_paths[0], npy_paths[0])),
                threading.Thread(target=capture_one, args=(self.visible_worker, dng_paths[1], npy_paths[1])),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if errors:
                raise RuntimeError("; ".join(errors))

            self.status.setText(
                "Saved raw Bayer pair: "
                f"{dng_paths[0].name}, {dng_paths[1].name} "
                "(plus matching .npy arrays)"
            )
        except Exception as exc:
            self._show_error(str(exc))


    def closeEvent(self, event) -> None:
        if not self.stop_voice_control():
            event.ignore()
            return
        self.stop_cameras()
        self.display.cleanup()
        event.accept()


def main() -> int:
    # Let Qt/Mesa choose the platform's native GL implementation. Explicitly
    # forcing a GLES context caused native crashes on some Raspberry Pi OS
    # PyQt5 builds. QOpenGLWidget will create the appropriate accelerated context.
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 820)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
