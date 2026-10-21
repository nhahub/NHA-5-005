from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from app import RestaurantOrderApplication, VoiceRuntime
from conversation import ConversationService
from order_db import tools
from voice import VoiceConfig
from voice import voice as audio


def test_fake_listen_conversation_database_and_speak(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "voice.sqlite3"))
    app = RestaurantOrderApplication()

    class Model:
        def complete(self, messages):
            return json.dumps(
                {
                    "intent": "add_item",
                    "items": [
                        {
                            "item_name": "Chicken Ranch",
                            "quantity": 2,
                            "size": "medium",
                            "modifiers": [],
                            "position": None,
                            "quantity_delta": None,
                        }
                    ],
                    "missing_information": [],
                }
            )

    heard = iter(["عايز اتنين Chicken Ranch وسط", "خلاص", "أيوه أكد"])
    spoken = []
    runtime = VoiceRuntime(ConversationService(app, Model()), lambda: next(heard), spoken.append)
    saved_id = runtime.run(app.start_session("voice-session"))
    assert saved_id is not None
    assert tools.get_saved_order(saved_id)["data"]["status"] == "confirmed"
    assert any("300.00" in message for message in spoken)
    assert str(saved_id) in spoken[-1]


def test_voice_runtime_silence_and_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "silence.sqlite3"))
    app = RestaurantOrderApplication()

    class NeverModel:
        def complete(self, messages):
            raise AssertionError("silence must not reach the LLM")

    service = ConversationService(app, NeverModel())
    spoken = []
    assert VoiceRuntime(service, lambda: "", spoken.append).run("s1") is None
    assert len(spoken) == 4
    assert VoiceRuntime(service, lambda: "خروج", spoken.append).run("s2") is None


def test_voice_env_consumes_all_advertised_settings(monkeypatch):
    values = {
        "STT_LANGUAGE": "en-US",
        "TTS_VOICE": "en-US-GuyNeural",
        "STT_TIMEOUT": "5",
        "STT_PHRASE_TIME_LIMIT": "9",
        "AMBIENT_NOISE_DURATION": "0.3",
        "PLAYBACK_VOLUME": "0.4",
        "TTS_RATE": "+10%",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    cfg = VoiceConfig.from_env()
    assert (cfg.stt_language, cfg.tts_voice) == ("en-US", "en-US-GuyNeural")
    assert (cfg.stt_timeout, cfg.stt_phrase_time_limit, cfg.ambient_noise_duration) == (5, 9, 0.3)
    assert cfg.playback_volume == 0.4 and cfg.tts_rate == "+10%"
    assert VoiceConfig().stt_language == "ar-EG"
    assert VoiceConfig().tts_voice == "ar-EG-SalmaNeural"


@pytest.fixture
def fake_recognition(monkeypatch):
    """Mock the submitted Google/microphone boundary, never real hardware/network."""

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    class WaitTimeoutError(Exception):
        pass

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Recognizer:
        def __init__(self):
            self.kwargs = None
            self.language = None
            self.error = None

        def adjust_for_ambient_noise(self, source, duration):
            self.duration = duration

        def listen(self, source, **kwargs):
            self.kwargs = kwargs
            return b"audio"

        def recognize_google(self, data, language):
            self.language = language
            if self.error:
                raise self.error
            return "  عايز بيتزا  "

    fake = SimpleNamespace(
        Recognizer=Recognizer,
        Microphone=Source,
        UnknownValueError=UnknownValueError,
        RequestError=RequestError,
        WaitTimeoutError=WaitTimeoutError,
        AudioData=lambda *args: args,
    )
    monkeypatch.setitem(sys.modules, "speech_recognition", fake)
    return fake


def test_submitted_stt_boundary_uses_egyptian_configuration(fake_recognition):
    listener = audio.SpeechListener(VoiceConfig(stt_timeout=4, stt_phrase_time_limit=7))
    assert listener.listen(show_prompt=False) == "عايز بيتزا"
    assert listener.recognizer.language == "ar-EG"
    assert listener.recognizer.kwargs == {"timeout": 4, "phrase_time_limit": 7}
    assert listener.recognizer.energy_threshold == 300
    assert listener.recognizer.dynamic_energy_threshold is True
    listener.recognizer.error = fake_recognition.UnknownValueError()
    assert listener.listen(show_prompt=False) == ""
    listener.recognizer.error = fake_recognition.RequestError()
    assert listener.transcribe_audio_data(b"silent") == ""


def test_tts_synthesis_and_playback_boundaries_mocked(monkeypatch):
    created = []

    class Communicate:
        def __init__(self, **kwargs):
            created.append(kwargs)

        async def stream(self):
            yield {"type": "metadata", "data": b"ignored"}
            yield {"type": "audio", "data": b"ID3audio"}

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=Communicate))
    played = []
    monkeypatch.setattr(audio.player, "play_bytes", lambda *args: played.append(args))
    cfg = VoiceConfig(tts_voice="salma", playback_volume=0.4)
    audio.speak("أهلا", config=cfg)
    assert created[0]["voice"] == "ar-EG-SalmaNeural"
    assert played == [(b"ID3audio", True, 0.4)]
    assert asyncio.run(audio.synthesize_to_bytes("")) == b""
    assert audio._resolve_voice("jenny") == "en-US-JennyNeural"


def test_audio_init_is_lazy_and_empty_speech_is_safe(monkeypatch):
    initialized = []
    music = SimpleNamespace(
        set_volume=lambda value: None,
        load=lambda data: None,
        play=lambda: None,
        get_busy=lambda: False,
    )
    mixer = SimpleNamespace(
        get_init=lambda: bool(initialized),
        init=lambda **kwargs: initialized.append(kwargs),
        music=music,
    )
    monkeypatch.setitem(sys.modules, "pygame", SimpleNamespace(mixer=mixer, error=RuntimeError))
    player = audio.AudioPlayer()
    assert initialized == []
    assert player.is_playing() is False
    player.play_bytes(b"mocked audio")
    assert len(initialized) == 1
    audio.speak("")
    assert len(initialized) == 1


@pytest.mark.parametrize("modern,prefix", [(True, ""), (True, "module."), (False, "")])
def test_kemetone_checkpoint_keys_load_strictly(monkeypatch, modern, prefix):
    """No Torch/models: verify exact adapter keys and non-affine identity terms."""

    class InstanceNorm:
        pass

    expected = {
        "conv.weight_g": "g",
        "conv.weight_v": "v",
        "norm.weight": "one",
        "norm.bias": "zero",
    }
    loaded = []

    def load(state, strict):
        assert strict is True
        assert state == expected
        loaded.append(state)

    layer = SimpleNamespace(
        state_dict=lambda: expected, get_submodule=lambda name: InstanceNorm(), load_state_dict=load
    )
    model = SimpleNamespace(decoder=layer, named_children=lambda: [("decoder", layer)])
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(InstanceNorm1d=InstanceNorm),
        ones_like=lambda value: "one",
        zeros_like=lambda value: "zero",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    keys = ("conv.parametrizations.weight.original0", "conv.parametrizations.weight.original1")
    if not modern:
        keys = ("conv.weight_g", "conv.weight_v")
    checkpoint = {"decoder": {prefix + keys[0]: "g", prefix + keys[1]: "v"}}
    audio.KemeToneSynthesizer._load_checkpoint(model, checkpoint)
    assert len(loaded) == 1


def test_kemetone_missing_learned_weights_fail_not_silent(monkeypatch):
    def load(state, strict):
        assert strict is True
        raise RuntimeError("Missing learned weight")

    layer = SimpleNamespace(
        state_dict=lambda: {"conv.weight_g": "g"},
        get_submodule=lambda name: object(),
        load_state_dict=load,
    )
    model = SimpleNamespace(decoder=layer, named_children=lambda: [("decoder", layer)])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(nn=SimpleNamespace(InstanceNorm1d=type("Norm", (), {}))),
    )
    with pytest.raises(RuntimeError, match="Missing learned"):
        audio.KemeToneSynthesizer._load_checkpoint(model, {"decoder": {}})


def test_kemetone_wrong_checkpoint_components_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    model = SimpleNamespace(named_children=lambda: [("decoder", object())])
    with pytest.raises(RuntimeError, match="components"):
        audio.KemeToneSynthesizer._load_checkpoint(model, {"unknown": {}})
