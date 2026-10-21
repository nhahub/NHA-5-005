"""Runtime configuration (env vars, optional .env file in the working dir)."""
import os
from pathlib import Path

CURRENCY = "EGP"


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def max_quantity() -> int:
    try:
        return max(1, int(os.environ.get("MAX_QUANTITY", "50")))
    except ValueError:
        return 50
