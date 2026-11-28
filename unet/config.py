import json
from pathlib import Path


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))
