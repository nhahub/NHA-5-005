"""Isolated offline F5-TTS worker using the validated Hag Shalaby recipe."""

import base64
import contextlib
import importlib.util
import inspect
import io
import json
import re
import sys
from pathlib import Path

REFERENCE_TEXT = "في سنة ألفين وخمسين، عشرة في المية، انتوا متخيلين يعني إيه عشرة في المية من الاقتصاد المصري"


def prepare_spoken_amounts(text):
    """Preserve EGP amounts while making decimals pronounceable by the normalizer."""
    def render(match):
        pounds, piasters = match.group(1), int(match.group(2))
        return f"{pounds} جنيه" + (f" و {piasters} قرش" if piasters else "")

    return re.sub(r"(?<![\d.])(\d+)\.(\d{2})\s*جنيه", render, text)


def main():
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import soundfile as sf
        import torch
        import torchaudio
        import yaml
        from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
        from f5_tts.model import DiT

        root = Path(sys.argv[1])
        device = sys.argv[2]
        required = ("model.pt", "config.yaml", "vocab.txt", "text_v3.py", "ref.wav")
        if not all((root / name).is_file() for name in required):
            raise RuntimeError("HAG_MODEL_PATH must contain all Hag Shalaby assets")

        def soundfile_load(path, *args, **kwargs):
            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data).T, rate

        torchaudio.load = soundfile_load
        spec = importlib.util.spec_from_file_location("hag_text_v3", root / "text_v3.py")
        normalizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normalizer)
        config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        arch = config["model"]["arch"]
        arch = {k: v for k, v in arch.items() if k in inspect.signature(DiT.__init__).parameters}
        model = load_model(DiT, arch, str(root / "model.pt"), "vocos",
                           str(root / "vocab.txt"), "euler", False, device)
        vocoder = load_vocoder("vocos", device=device)
        for line in sys.stdin:
            try:
                text = json.loads(line)["text"]
                torch.manual_seed(42)
                wave, rate, _ = infer_process(
                    str(root / "ref.wav"), REFERENCE_TEXT,
                    normalizer.normalize_v3(prepare_spoken_amounts(text)),
                    model, vocoder, "vocos", nfe_step=32, cfg_strength=2.0, device=device,
                )
                stream = io.BytesIO()
                sf.write(stream, wave, rate, format="WAV", subtype="PCM_16")
                response = {"wav": base64.b64encode(stream.getvalue()).decode("ascii")}
            except Exception as exc:  # noqa: BLE001 - report inference errors through the pipe
                response = {"error": str(exc)}
            print(json.dumps(response), file=protocol, flush=True)


if __name__ == "__main__":
    main()
