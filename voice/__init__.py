"""Microphone STT and speaker TTS; importing never opens audio devices."""

from .voice import (
    AVAILABLE_VOICES,
    DEFAULT_CONFIG,
    AudioPlayer,
    SpeechListener,
    VoiceConfig,
    calibrate_mic,
    is_speaking,
    listen,
    speak,
    speak_async,
    stop_speaking,
    synthesize_to_bytes,
    transcribe_file,
)

__all__ = [
    "AVAILABLE_VOICES",
    "DEFAULT_CONFIG",
    "AudioPlayer",
    "SpeechListener",
    "VoiceConfig",
    "calibrate_mic",
    "is_speaking",
    "listen",
    "speak",
    "speak_async",
    "stop_speaking",
    "synthesize_to_bytes",
    "transcribe_file",
]
