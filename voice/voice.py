"""Lazy local Whisper/KemeTone, optional Google/Edge, and pygame playback.

Optional SDKs and audio devices are opened only when used. This module transports
speech and text; it has no order or conversation logic.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

AVAILABLE_VOICES = {
    "en_us_female": "en-US-JennyNeural",
    "en_us_male": "en-US-GuyNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "aria": "en-US-AriaNeural",
    "christopher": "en-US-ChristopherNeural",
    "en_uk_female": "en-GB-SoniaNeural",
    "en_uk_male": "en-GB-RyanNeural",
    "sonia": "en-GB-SoniaNeural",
    "ryan": "en-GB-RyanNeural",
    "salma": "ar-EG-SalmaNeural",
    "shakir": "ar-EG-ShakirNeural",
    "zariyah": "ar-SA-ZariyahNeural",
    "hamed": "ar-SA-HamedNeural",
}


@dataclass
class VoiceConfig:
    # Explicit legacy configs remain compatible; runtime from_env defaults to local.
    stt_provider: str = "google"
    tts_provider: str = "edge"
    stt_model_path: str = ""
    tts_model_path: str = ""
    hag_python: str = ""
    hag_model_path: str = ""
    hag_cache_path: str = ""
    hag_device: str = "cuda"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8_float16"
    tts_speed: float = 1.0
    stt_language: str = "ar-EG"
    stt_timeout: float = 8.0
    stt_phrase_time_limit: float = 15.0
    ambient_noise_duration: float = 0.6
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    tts_voice: str = "ar-EG-SalmaNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    playback_volume: float = 1.0

    @classmethod
    def from_env(cls) -> VoiceConfig:
        return cls(
            stt_provider=os.getenv("STT_PROVIDER", "faster_whisper"),
            tts_provider=os.getenv("TTS_PROVIDER", "kemetone"),
            stt_model_path=os.getenv("LOCAL_STT_MODEL_PATH", ""),
            tts_model_path=os.getenv("LOCAL_TTS_MODEL_PATH", ""),
            hag_python=os.getenv("HAG_PYTHON", ""),
            hag_model_path=os.getenv("HAG_MODEL_PATH", ""),
            hag_cache_path=os.getenv("HAG_HF_CACHE", ""),
            hag_device=os.getenv("HAG_DEVICE", "cuda"),
            stt_device=os.getenv("STT_DEVICE", "cuda"),
            stt_compute_type=os.getenv(
                "STT_COMPUTE_TYPE", "int8" if os.getenv("STT_DEVICE") == "cpu" else "int8_float16"
            ),
            tts_speed=float(os.getenv("TTS_SPEED", "1.0")),
            stt_language=os.getenv("STT_LANGUAGE", "ar-EG"),
            tts_voice=os.getenv("TTS_VOICE", "ar-EG-SalmaNeural"),
            stt_timeout=float(os.getenv("STT_TIMEOUT", "8.0")),
            stt_phrase_time_limit=float(os.getenv("STT_PHRASE_TIME_LIMIT", "15.0")),
            ambient_noise_duration=float(os.getenv("AMBIENT_NOISE_DURATION", "0.6")),
            playback_volume=float(os.getenv("PLAYBACK_VOLUME", "1.0")),
            tts_rate=os.getenv("TTS_RATE", "+0%"),
        )


DEFAULT_CONFIG = VoiceConfig.from_env()


class AudioPlayer:
    """Same in-memory playback behavior, with lazy pygame/audio initialization."""

    def __init__(self):
        self._lock = threading.Lock()

    def play_bytes(self, audio_bytes: bytes, wait=True, volume=1.0):
        if not audio_bytes:
            return
        import pygame

        with self._lock:
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)
                except pygame.error:
                    pygame.mixer.init()
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
            self._stream = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(self._stream)
            pygame.mixer.music.play()
        if wait:
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

    def stop(self):
        pygame = sys.modules.get("pygame")
        if pygame and pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def is_playing(self):
        pygame = sys.modules.get("pygame")
        return bool(pygame and pygame.mixer.get_init() and pygame.mixer.music.get_busy())

    def close(self):
        pygame = sys.modules.get("pygame")
        if pygame and pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()


player = AudioPlayer()


class SpeechListener:
    def __init__(self, config=None):
        import speech_recognition as sr

        self.config = config or DEFAULT_CONFIG
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.config.energy_threshold
        self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.5
        self._calibrated = False

    def calibrate_noise(self, duration=None):
        import speech_recognition as sr

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=duration if duration is not None else self.config.ambient_noise_duration,
            )
        self._calibrated = True

    def listen(self, timeout=None, phrase_time_limit=None, language=None, show_prompt=True):
        import speech_recognition as sr

        try:
            with sr.Microphone() as source:
                if not self._calibrated:
                    self.recognizer.adjust_for_ambient_noise(
                        source, duration=self.config.ambient_noise_duration
                    )
                    self._calibrated = True
                if show_prompt:
                    print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout if timeout is not None else self.config.stt_timeout,
                    phrase_time_limit=(
                        phrase_time_limit
                        if phrase_time_limit is not None
                        else self.config.stt_phrase_time_limit
                    ),
                )
                return self.recognizer.recognize_google(
                    audio, language=language or self.config.stt_language
                ).strip()
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return ""
        except sr.RequestError:
            if show_prompt:
                print("Speech recognition service unavailable.")
            return ""
        except Exception:  # noqa: BLE001 - audio-driver/platform errors are contained
            if show_prompt:
                print("Microphone unavailable.")
            return ""

    def transcribe_audio_data(self, audio_data, language=None, sample_rate=16000, sample_width=2):
        import speech_recognition as sr

        audio = (
            sr.AudioData(audio_data, sample_rate, sample_width)
            if isinstance(audio_data, bytes)
            else audio_data
        )
        try:
            return self.recognizer.recognize_google(
                audio, language=language or self.config.stt_language
            ).strip()
        except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError):
            return ""


class LocalWhisperListener(SpeechListener):
    """Microphone capture is shared; recognition never contacts Google."""

    def __init__(self, config=None):
        # No microphone, recognizer, CUDA or weights at construction time.
        self.config = config or DEFAULT_CONFIG
        self._model = None
        self._calibrated = False

    def transcribe_file(self, source, language=None):
        if self._model is None:
            root = Path(self.config.stt_model_path)
            if not self.config.stt_model_path or not (root / "model.bin").is_file():
                raise RuntimeError("LOCAL_STT_MODEL_PATH must reference existing CT2 weights")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                str(root),
                device=self.config.stt_device,
                compute_type=self.config.stt_compute_type,
                local_files_only=True,
                num_workers=1,
            )
        segments, _ = self._model.transcribe(
            source,
            language=(language or self.config.stt_language).split("-")[0],
            task="transcribe",
            beam_size=1,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def calibrate_noise(self, duration=None):
        import speech_recognition as sr

        self.recognizer = sr.Recognizer()
        super().calibrate_noise(duration)

    def listen(self, timeout=None, phrase_time_limit=None, language=None, show_prompt=True):
        import numpy as np
        import speech_recognition as sr

        # Load/driver errors deliberately surface, rather than masquerading as silence.
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.config.energy_threshold
        recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
        try:
            with sr.Microphone() as source:
                if not self._calibrated:
                    recognizer.adjust_for_ambient_noise(
                        source, duration=self.config.ambient_noise_duration
                    )
                    self._calibrated = True
                if show_prompt:
                    print("Listening...")
                audio = recognizer.listen(
                    source,
                    timeout=timeout if timeout is not None else self.config.stt_timeout,
                    phrase_time_limit=phrase_time_limit
                    if phrase_time_limit is not None
                    else self.config.stt_phrase_time_limit,
                )
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return ""
        pcm = (
            np.frombuffer(
                audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )
        return self.transcribe_file(pcm, language)


class KemeToneSynthesizer:
    """Use only existing local weights/frontend; CPU inference and in-memory WAV."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self._model = None

    def _load(self):
        root = Path(self.config.tts_model_path)
        required = [
            root / "config.json",
            root / "kemetone.pth",
            root / "voices/kemetone.pt",
            root / "kemetone/__init__.py",
        ]
        if not self.config.tts_model_path or not all(path.is_file() for path in required):
            raise RuntimeError("LOCAL_TTS_MODEL_PATH must reference complete KemeTone assets")
        # Explicit local weights and offline flags prevent implicit Hugging Face downloads.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from kokoro import KModel

        name = "_restaurant_kemetone"
        spec = importlib.util.spec_from_file_location(
            name,
            root / "kemetone/__init__.py",
            submodule_search_locations=[str(root / "kemetone")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # External frontend files are read-only assets, including their cache paths.
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
            self._arabic = module.EgyptianG2P()
        finally:
            sys.dont_write_bytecode = previous_bytecode
        from misaki.espeak import EspeakG2P

        self._english = EspeakG2P(language="en-us")
        model = (
            KModel(
                repo_id="local/kemetone",
                config=str(root / "config.json"),
                model=str(root / "kemetone.pth"),
            )
            .to("cpu")
            .eval()
        )
        # Kokoro 0.9.4 silently strips unprefixed keys after any load failure.
        # KemeTone uses modern weight_norm keys: reload explicitly and strictly.
        checkpoint = torch.load(root / "kemetone.pth", map_location="cpu", weights_only=True)
        self._load_checkpoint(model, checkpoint)
        self._voice = torch.load(root / "voices/kemetone.pt", map_location="cpu", weights_only=True)
        self._model = model

    @staticmethod
    def _load_checkpoint(model, checkpoint):
        """Adapt serialization only; reject missing learned weights instead of buzzing."""
        import torch

        if set(checkpoint) != {name for name, _ in model.named_children()}:
            raise RuntimeError("KemeTone checkpoint components do not match the runtime")
        for name, weights in checkpoint.items():
            layer = getattr(model, name)
            mapped = {}
            strip_prefix = bool(weights) and all(key.startswith("module.") for key in weights)
            for key, value in weights.items():
                key = key[7:] if strip_prefix else key
                key = key.replace(".parametrizations.weight.original0", ".weight_g")
                key = key.replace(".parametrizations.weight.original1", ".weight_v")
                if key in mapped:
                    raise RuntimeError("Duplicate KemeTone checkpoint parameter")
                mapped[key] = value
            expected = layer.state_dict()
            supplied = set(mapped)
            for key in expected.keys() - mapped.keys():
                parent, attribute = key.rsplit(".", 1)
                normalization = layer.get_submodule(parent)
                # This runtime added affine InstanceNorm for ONNX export. The
                # checkpoint used non-affine normalization: identity is equivalent.
                if (
                    isinstance(normalization, torch.nn.InstanceNorm1d)
                    and attribute in {"weight", "bias"}
                    and parent + ".weight" not in supplied
                    and parent + ".bias" not in supplied
                ):
                    mapped[key] = (
                        torch.ones_like(expected[key])
                        if attribute == "weight"
                        else torch.zeros_like(expected[key])
                    )
            layer.load_state_dict(mapped, strict=True)

    def synthesize(self, text):
        if self._model is None:
            self._load()
        import soundfile as sf
        import torch

        # KemeTone's Arabic normalizer drops Latin words: phonemize those separately.
        chunks = re.split(r"([A-Za-z][A-Za-z '\-]*)", text)
        phonemes = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            value = self._english(chunk) if re.search("[A-Za-z]", chunk) else self._arabic(chunk)
            phonemes.append(value[0] if isinstance(value, tuple) else value)
        stream = io.BytesIO()
        waves = []
        # Kokoro supports at most 510 phonemes per inference; never truncate a summary.
        joined = " ".join(phonemes)
        for offset in range(0, len(joined), 500):
            piece = joined[offset : offset + 500]
            with torch.inference_mode():
                audio = self._model(piece, self._voice[len(piece) - 1], speed=self.config.tts_speed)
            waves.append(audio.detach().cpu().numpy())
        if not waves:
            return b""
        import numpy as np

        sf.write(stream, np.concatenate(waves), 24000, format="WAV", subtype="PCM_16")
        return stream.getvalue()


class HagShalabySynthesizer:
    """Keep the benchmark's isolated F5 runtime warm; exchange WAVs over local pipes."""

    def __init__(self, config):
        self.config = config
        self._process = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def synthesize(self, text):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                if not Path(self.config.hag_python).is_file():
                    raise RuntimeError("HAG_PYTHON must point to the isolated Python executable")
                worker = Path(__file__).with_name("hag_worker.py")
                env = os.environ.copy()
                env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
                if self.config.hag_cache_path:
                    env["HF_HOME"] = self.config.hag_cache_path
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                self._process = subprocess.Popen(
                    [self.config.hag_python, "-u", str(worker), self.config.hag_model_path,
                     self.config.hag_device],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                    encoding="utf-8", env=env, creationflags=flags,
                )
            try:
                self._process.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("Hag Shalaby worker exited unexpectedly")
                result = json.loads(line)
                if "error" in result:
                    raise RuntimeError("Hag Shalaby synthesis failed: " + result["error"])
                return base64.b64decode(result["wav"], validate=True)
            except (BrokenPipeError, ValueError, KeyError) as exc:
                self.close()
                raise RuntimeError("Hag Shalaby worker communication failed") from exc

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()


_default_listener = None
_default_synthesizer = None


def listen(timeout=None, phrase_time_limit=None, language=None, show_prompt=True, config=None):
    global _default_listener
    cfg = config or DEFAULT_CONFIG
    if cfg.stt_provider not in {"google", "faster_whisper"}:
        raise ValueError("Unknown STT_PROVIDER")
    listener_class = (
        LocalWhisperListener if cfg.stt_provider == "faster_whisper" else SpeechListener
    )
    if config is not None:
        listener = listener_class(config)
    else:
        if _default_listener is None:
            _default_listener = listener_class()
        listener = _default_listener
    return listener.listen(timeout, phrase_time_limit, language, show_prompt)


def calibrate_mic(duration=1.0):
    global _default_listener
    if _default_listener is None:
        cls = (
            LocalWhisperListener
            if DEFAULT_CONFIG.stt_provider == "faster_whisper"
            else SpeechListener
        )
        _default_listener = cls()
    _default_listener.calibrate_noise(duration)


def transcribe_file(file_path, language=None, config=None):
    cfg = config or DEFAULT_CONFIG
    if cfg.stt_provider == "faster_whisper":
        return LocalWhisperListener(cfg).transcribe_file(file_path, language)
    if cfg.stt_provider != "google":
        raise ValueError("Unknown STT_PROVIDER")
    import speech_recognition as sr

    listener = SpeechListener(cfg)
    with sr.AudioFile(file_path) as source:
        audio = listener.recognizer.record(source)
    return listener.transcribe_audio_data(audio, language)


def _resolve_voice(voice_name):
    name = voice_name or DEFAULT_CONFIG.tts_voice
    return AVAILABLE_VOICES.get(name.lower().strip(), name)


async def synthesize_to_bytes(text, voice=None, rate=None, pitch=None, volume=None, config=None):
    global _default_synthesizer
    if not text or not text.strip():
        return b""
    cfg = config or DEFAULT_CONFIG
    if cfg.tts_provider in {"kemetone", "hag_shalaby"}:
        if _default_synthesizer is None or _default_synthesizer.config != cfg:
            if isinstance(_default_synthesizer, HagShalabySynthesizer):
                _default_synthesizer.close()
            cls = HagShalabySynthesizer if cfg.tts_provider == "hag_shalaby" else KemeToneSynthesizer
            _default_synthesizer = cls(cfg)
        return await asyncio.to_thread(_default_synthesizer.synthesize, text.strip())
    if cfg.tts_provider != "edge":
        raise ValueError("Unknown TTS_PROVIDER")
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text.strip(),
        voice=_resolve_voice(voice or cfg.tts_voice),
        rate=rate or cfg.tts_rate,
        pitch=pitch or cfg.tts_pitch,
        volume=volume or cfg.tts_volume,
    )
    audio_chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.extend(chunk["data"])
    return bytes(audio_chunks)


async def speak_async(text, voice=None, wait=True, rate=None, pitch=None, volume=None, config=None):
    cfg = config or DEFAULT_CONFIG
    audio = await synthesize_to_bytes(text, voice, rate, pitch, volume, cfg)
    if audio:
        await asyncio.to_thread(player.play_bytes, audio, wait, cfg.playback_volume)


def speak(text, voice=None, wait=True, rate=None, pitch=None, volume=None, config=None):
    if not text or not text.strip():
        return
    coroutine = speak_async(text, voice, wait, rate, pitch, volume, config)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(asyncio.run, coroutine).result()


def stop_speaking():
    player.stop()


def is_speaking():
    return player.is_playing()
