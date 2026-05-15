from __future__ import annotations

import platform
import shutil
import subprocess
import time
from pathlib import Path

from .config import Settings
from .models import BotProfile


class ConsoleSpeaker:
    status = "TTS: disabled; console text only."

    def speak(self, participant: str, text: str, profile: BotProfile | None = None) -> None:
        print(f"{participant}: {text}")


class KokoroSpeaker(ConsoleSpeaker):
    def __init__(self, model_path: str, voices_path: str) -> None:
        self._kokoro = None
        self._sounddevice = None
        self._load_error: str | None = None
        model_path = _resolve_kokoro_model_path(model_path)

        try:
            from kokoro_onnx import Kokoro
            import sounddevice
        except ImportError as exc:
            self._load_error = str(exc)
            return

        if not Path(model_path).exists() or not Path(voices_path).exists():
            self._load_error = (
                f"Missing Kokoro model or voices file: {model_path}, {voices_path}"
            )
            return

        try:
            self._kokoro = Kokoro(model_path, voices_path)
            self._sounddevice = sounddevice
        except Exception as exc:  # pragma: no cover - depends on local audio stack.
            self._load_error = str(exc)

    @property
    def is_available(self) -> bool:
        return self._kokoro is not None and self._sounddevice is not None

    @property
    def status(self) -> str:
        if self.is_available:
            return "TTS: enabled with kokoro-onnx."
        if self._load_error:
            return f"TTS: kokoro-onnx unavailable; {self._load_error}"
        return "TTS: kokoro-onnx unavailable."

    def speak(self, participant: str, text: str, profile: BotProfile | None = None) -> None:
        super().speak(participant, text, profile)
        if not self._kokoro or not self._sounddevice:
            if self._load_error:
                print(f"[tts disabled] {self._load_error}")
                self._load_error = None
            return

        voice = profile.voice if profile else "af_bella"
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=voice,
                speed=1.0,
                lang="en-us",
            )
            self._sounddevice.play(samples, sample_rate)
            self._sounddevice.wait()
        except Exception as exc:  # pragma: no cover - depends on local audio stack.
            print(f"[tts failed] {exc}")


class MacOSSaySpeaker(ConsoleSpeaker):
    def __init__(self, voice: str) -> None:
        self._voice = voice.strip()

    @property
    def status(self) -> str:
        if self._voice:
            return f"TTS: enabled with macOS say fallback voice '{self._voice}'."
        return "TTS: enabled with macOS say fallback."

    def speak(self, participant: str, text: str, profile: BotProfile | None = None) -> None:
        super().speak(participant, text, profile)
        try:
            command = ["say"]
            if self._voice:
                command.extend(["-v", self._voice])
            command.append(text)
            subprocess.run(command, check=False)
        except Exception as exc:  # pragma: no cover - depends on local OS audio stack.
            print(f"[tts failed] {exc}")


class ConsoleListener:
    status = "Voice input: disabled; keyboard input only."

    def listen(self) -> str:
        return input("> ").strip()


class WhisperVadListener(ConsoleListener):
    def __init__(
        self,
        whisper_model_name: str,
        silence_ms: int,
        sample_rate: int = 16_000,
        block_size: int = 512,
        max_seconds: int = 30,
    ) -> None:
        self._fallback = False
        self._load_error: str | None = None
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._max_seconds = max_seconds
        self._silence_ms = silence_ms

        try:
            import numpy as np
            import sounddevice as sd
            import torch
            from faster_whisper import WhisperModel
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as exc:
            self._fallback = True
            self._load_error = str(exc)
            return

        self._np = np
        self._sd = sd
        self._torch = torch
        self._whisper = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")
        self._vad_iterator = VADIterator(
            load_silero_vad(),
            sampling_rate=sample_rate,
            min_silence_duration_ms=silence_ms,
        )

    @property
    def status(self) -> str:
        if self._fallback and self._load_error:
            return f"Voice input: unavailable; {self._load_error}"
        if self._fallback:
            return "Voice input: unavailable; keyboard input only."
        return (
            "Voice input: enabled with Silero VAD and faster-whisper "
            f"(max {self._max_seconds}s, silence {self._silence_ms}ms)."
        )

    def listen(self) -> str:
        if self._fallback:
            if self._load_error:
                print(f"[voice input disabled] {self._load_error}")
                self._load_error = None
            return super().listen()

        print("[listening]")
        try:
            audio = self._record_until_silence()
            segments, _info = self._whisper.transcribe(
                audio,
                language="en",
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:  # pragma: no cover - depends on local audio stack.
            print(f"[voice input failed] {exc}")
            return super().listen()

        if not text:
            print("[heard nothing; type the command]")
            return super().listen()

        print(f"[heard] {text}")
        return text

    def _record_until_silence(self):
        chunks = []
        started = False
        start_time = time.monotonic()
        self._vad_iterator.reset_states()

        with self._sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._block_size,
        ) as stream:
            while time.monotonic() - start_time < self._max_seconds:
                data, _overflowed = stream.read(self._block_size)
                chunk = data[:, 0].copy()
                event = self._vad_iterator(self._torch.from_numpy(chunk))

                if event and "start" in event:
                    started = True
                if started:
                    chunks.append(chunk)
                if event and "end" in event and chunks:
                    break

        if not chunks:
            return self._np.zeros(0, dtype="float32")
        return self._np.concatenate(chunks).astype("float32")


def build_speaker(settings: Settings) -> ConsoleSpeaker:
    if not settings.enable_tts:
        return ConsoleSpeaker()

    backend = settings.tts_backend.strip().lower()
    if backend in {"auto", "kokoro"}:
        kokoro = KokoroSpeaker(settings.kokoro_model_path, settings.kokoro_voices_path)
        if kokoro.is_available or backend == "kokoro":
            return kokoro

    if backend in {"auto", "macos", "say"}:
        if platform.system() == "Darwin" and shutil.which("say"):
            return MacOSSaySpeaker(settings.macos_say_voice)

    return ConsoleSpeaker()


def _resolve_kokoro_model_path(model_path: str) -> str:
    configured_path = Path(model_path)
    if configured_path.exists():
        return model_path

    if configured_path.name == "kokoro-v1.0.onnx":
        int8_path = configured_path.with_name("kokoro-v1.0.int8.onnx")
        if int8_path.exists():
            return str(int8_path)

    return model_path


def build_listener(settings: Settings) -> ConsoleListener:
    if settings.enable_voice_input:
        return WhisperVadListener(
            settings.whisper_model,
            settings.vad_silence_ms,
            max_seconds=settings.voice_max_seconds,
        )
    return ConsoleListener()
