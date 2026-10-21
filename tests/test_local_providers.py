"""Local boundaries are tested without weights, servers, CUDA or audio devices."""

import io
import json
import sys
from types import SimpleNamespace

import pytest

from app import RestaurantOrderApplication, VoiceRuntime, main
from conversation import ConversationService, LocalUnderstandingProvider, configured_provider
from voice import voice

VALID = json.dumps(
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


@pytest.mark.parametrize("provider", ["llamacpp", "ollama"])
def test_local_http_pipeline(provider, monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8765")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:8766")
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        data = (
            {"choices": [{"message": {"content": VALID}}]}
            if provider == "llamacpp"
            else {"message": {"content": VALID}}
        )
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *args: SimpleNamespace(open=open_request)
    )
    app = RestaurantOrderApplication(tmp_path / "local.sqlite3")
    service = ConversationService(app, configured_provider())
    assert service.handle("s", "عايز اتنين Chicken Ranch وسط").results[0].outcome == "item_added"
    assert requests[0].full_url == (
        "http://127.0.0.1:8765/v1/chat/completions"
        if provider == "llamacpp"
        else "http://127.0.0.1:8766/api/chat"
    )
    body = json.loads(requests[0].data)
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "CURRENT DATABASE DRAFT" in body["messages"][0]["content"]
    assert not service.handle("s", "خلاص").completed
    assert service.handle("s", "أيوه أكد").completed
    spoken = []
    utterances = iter(["عايز اتنين Chicken Ranch وسط", "خلاص", "أيوه أكد"])
    runtime = VoiceRuntime(service, lambda: next(utterances), spoken.append)
    assert isinstance(runtime.run("voice-local"), int)
    assert "300.00" in spoken[-2]


def test_local_bad_output_is_validated(monkeypatch, tmp_path):
    monkeypatch.setattr(LocalUnderstandingProvider, "complete", lambda *args: '{"intent":"hack"}')
    app = RestaurantOrderApplication(tmp_path / "invalid.sqlite3")
    reply = ConversationService(app, LocalUnderstandingProvider()).handle("s", "hello")
    assert not reply.results
    assert app.current_order("s")["data"]["items"] == []


def test_loopback_only(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://192.168.1.2:8080")
    with pytest.raises(ValueError, match="loopback"):
        LocalUnderstandingProvider()


def test_local_defaults_no_key(monkeypatch):
    for key in ("GROQ_API_KEY", "LLM_PROVIDER", "STT_PROVIDER", "TTS_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    assert isinstance(configured_provider(), LocalUnderstandingProvider)
    cfg = voice.VoiceConfig.from_env()
    assert (cfg.stt_provider, cfg.tts_provider) == ("faster_whisper", "kemetone")


def test_whisper_lazy_and_local_only(monkeypatch, tmp_path):
    (tmp_path / "model.bin").touch()
    calls = []

    def model(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            transcribe=lambda source, **opts: ([SimpleNamespace(text=" تشيكن رانش ")], None)
        )

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model))
    listener = voice.LocalWhisperListener(
        voice.VoiceConfig(stt_model_path=str(tmp_path), stt_device="cpu", stt_compute_type="int8")
    )
    assert listener._model is None and not calls
    assert listener.transcribe_file("sample.wav") == "تشيكن رانش"
    assert listener.transcribe_file("sample.wav") == "تشيكن رانش"
    assert len(calls) == 1 and calls[0]["local_files_only"] is True


def test_kemetone_lazy_dispatch(monkeypatch):
    cfg = voice.VoiceConfig(tts_provider="kemetone")
    synth = voice.KemeToneSynthesizer(cfg)
    assert synth._model is None
    monkeypatch.setattr(voice, "_default_synthesizer", None)
    monkeypatch.setattr(voice.KemeToneSynthesizer, "synthesize", lambda self, text: b"wav")
    spoken = []
    monkeypatch.setattr(voice.player, "play_bytes", lambda data, *args: spoken.append(data))
    voice.speak("أهلا", config=cfg)
    assert spoken == [b"wav"]


def test_text_entry_does_not_import_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["app.py", "--mode", "text", "--db-path", str(tmp_path / "text.sqlite3")]
    )
    monkeypatch.setattr("builtins.input", lambda *args: "exit")
    monkeypatch.setattr(
        voice.LocalWhisperListener,
        "__init__",
        lambda *args: pytest.fail("text mode initialized STT"),
    )
    monkeypatch.setattr(
        voice.KemeToneSynthesizer,
        "__init__",
        lambda *args: pytest.fail("text mode initialized TTS"),
    )
    main()
