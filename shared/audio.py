from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Protocol

from .paths import resolve_repo_path


class SpeechProfile(Protocol):
    voice: str


class AudioSettings(Protocol):
    enable_tts: bool
    tts_backend: str
    macos_say_voice: str
    enable_voice_input: bool
    enable_audio_cues: bool
    audio_cue_volume: float
    kokoro_model_path: str
    kokoro_voices_path: str
    whisper_model: str
    vad_silence_ms: int
    voice_max_seconds: int


class AudioCuePlayer:
    def __init__(self, enabled: bool, volume: float = 0.2, sample_rate: int = 22_050) -> None:
        self._enabled = enabled
        self._volume = max(0.0, min(volume, 1.0))
        self._sample_rate = sample_rate
        self._load_error: str | None = None

        if not enabled:
            self._np = None
            self._sd = None
            return

        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            self._np = None
            self._sd = None
            self._load_error = str(exc)
            return

        self._np = np
        self._sd = sd

    @property
    def status(self) -> str:
        if not self._enabled:
            return "Audio cues: disabled."
        if self._load_error:
            return f"Audio cues: unavailable; {self._load_error}"
        return "Audio cues: enabled."

    def command_accepted(self) -> None:
        self._play_sequence([(880, 0.055), (1175, 0.075)])

    def command_rejected(self) -> None:
        self._play_sequence([(220, 0.11), (185, 0.14)])

    def user_turn_started(self) -> None:
        self._play_sequence([(523, 0.08), (784, 0.08), (1047, 0.1)])

    def _play_sequence(self, tones: list[tuple[float, float]]) -> None:
        if not self._enabled or self._np is None or self._sd is None:
            return

        try:
            chunks = [
                self._tone(frequency=frequency, duration=duration)
                for frequency, duration in tones
            ]
            pause = self._np.zeros(int(self._sample_rate * 0.025), dtype="float32")
            samples = pause
            for chunk in chunks:
                samples = self._np.concatenate((samples, chunk, pause))

            self._sd.play(samples, self._sample_rate)
            self._sd.wait()
        except Exception as exc:  # pragma: no cover - depends on local audio stack.
            self._load_error = str(exc)

    def _tone(self, frequency: float, duration: float):
        assert self._np is not None
        sample_count = int(self._sample_rate * duration)
        t = self._np.linspace(0, duration, sample_count, endpoint=False)
        envelope = self._np.sin(self._np.linspace(0, self._np.pi, sample_count))
        samples = self._np.sin(2 * self._np.pi * frequency * t) * envelope
        return (samples * self._volume).astype("float32")


class ConsoleSpeaker:
    status = "TTS: disabled; console text only."

    def speak(
        self,
        participant: str,
        text: str,
        profile: SpeechProfile | None = None,
    ) -> None:
        print(f"{participant}: {text}")


class KokoroSpeaker(ConsoleSpeaker):
    def __init__(self, model_path: str, voices_path: str) -> None:
        self._kokoro = None
        self._sounddevice = None
        self._load_error: str | None = None
        model_path = _resolve_kokoro_model_path(model_path)
        voices_path = str(resolve_repo_path(voices_path))

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

    def speak(
        self,
        participant: str,
        text: str,
        profile: SpeechProfile | None = None,
    ) -> None:
        super().speak(participant, text, profile)
        if not self._kokoro or not self._sounddevice:
            if self._load_error:
                print(f"[tts disabled] {self._load_error}")
                self._load_error = None
            return

        voice = profile.voice if profile else "af_bella"
        spoken_text = prepare_tts_text(text)
        try:
            samples, sample_rate = self._kokoro.create(
                spoken_text,
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

    def speak(
        self,
        participant: str,
        text: str,
        profile: SpeechProfile | None = None,
    ) -> None:
        super().speak(participant, text, profile)
        try:
            command = ["say"]
            if self._voice:
                command.extend(["-v", self._voice])
            command.append(prepare_tts_text(text))
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
        cue_player: AudioCuePlayer,
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
        self._cue_player = cue_player

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
            audio, _stop_reason = self._record_until_silence()
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
        stop_reason = "timeout"
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
                    stop_reason = "silence"
                    break

        if not chunks:
            return self._np.zeros(0, dtype="float32"), stop_reason
        return self._np.concatenate(chunks).astype("float32"), stop_reason


def build_speaker(settings: AudioSettings) -> ConsoleSpeaker:
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
    configured_path = resolve_repo_path(model_path)
    if configured_path.exists():
        return str(configured_path)

    if configured_path.name == "kokoro-v1.0.onnx":
        int8_path = configured_path.with_name("kokoro-v1.0.int8.onnx")
        if int8_path.exists():
            return str(int8_path)

    return str(configured_path)


def prepare_tts_text(text: str) -> str:
    return re.sub(
        r"(?<![\w.])(\d+)\.(\d+)(?!\w|\.\d)",
        _decimal_to_spoken_text,
        text,
    )


def _decimal_to_spoken_text(match: re.Match[str]) -> str:
    integer_part = match.group(1)
    decimal_part = " ".join(match.group(2))
    return f"{integer_part} point {decimal_part}"


def build_audio_cues(settings: AudioSettings) -> AudioCuePlayer:
    return AudioCuePlayer(
        enabled=settings.enable_audio_cues,
        volume=settings.audio_cue_volume,
    )


def build_listener(settings: AudioSettings, cue_player: AudioCuePlayer) -> ConsoleListener:
    if settings.enable_voice_input:
        return WhisperVadListener(
            settings.whisper_model,
            settings.vad_silence_ms,
            cue_player=cue_player,
            max_seconds=settings.voice_max_seconds,
        )
    return ConsoleListener()
