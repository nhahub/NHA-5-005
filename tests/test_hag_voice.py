"""The isolated TTS boundary is tested without models or audio devices."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

from voice import voice
from voice.hag_worker import prepare_spoken_amounts


def test_hag_worker_is_lazy_offline_and_reused(monkeypatch, tmp_path):
    python = tmp_path / "python.exe"
    python.touch()
    calls = []
    incoming = io.StringIO()
    outgoing = io.StringIO("\n".join([json.dumps({"wav": base64.b64encode(b"wav").decode()})] * 2) + "\n")
    process = SimpleNamespace(stdin=incoming, stdout=outgoing, poll=lambda: None, wait=lambda timeout: 0)

    def start(args, **kwargs):
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(voice.subprocess, "Popen", start)
    cfg = voice.VoiceConfig(tts_provider="hag_shalaby", hag_python=str(python), hag_model_path="model")
    synth = voice.HagShalabySynthesizer(cfg)
    assert not calls
    assert synth.synthesize("أهلا") == b"wav"
    assert synth.synthesize("تمام") == b"wav"
    assert len(calls) == 1
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert json.loads(incoming.getvalue().splitlines()[0])["text"] == "أهلا"
    synth.close()


def test_hag_dispatch(monkeypatch):
    cfg = voice.VoiceConfig(tts_provider="hag_shalaby")
    monkeypatch.setattr(voice, "_default_synthesizer", None)
    monkeypatch.setattr(voice.HagShalabySynthesizer, "synthesize", lambda self, text: b"wav")
    assert asyncio.run(voice.synthesize_to_bytes("أهلا", config=cfg)) == b"wav"


def test_hag_spoken_amounts_preserve_pounds_and_piasters():
    assert prepare_spoken_amounts("الإجمالي 300.00 جنيه") == "الإجمالي 300 جنيه"
    assert prepare_spoken_amounts("الإجمالي 150.50 جنيه") == "الإجمالي 150 جنيه و 50 قرش"
    assert prepare_spoken_amounts("أهلاً بيك") == "أهلاً بيك"
