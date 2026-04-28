"""Configuration loader."""
import yaml
from pathlib import Path

_config = None

def load_config(path: str = None) -> dict:
    global _config
    if _config is not None and path is None:
        return _config

    if path is None:
        path = Path(__file__).parent / "config.yaml"
        if not path.exists():
            path = Path(__file__).parent / "config.example.yaml"

    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config

def get_config() -> dict:
    if _config is None:
        return load_config()
    return _config
