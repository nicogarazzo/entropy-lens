"""Auto-detect architecture from config.json and return the correct extractor."""

import json
from pathlib import Path

from .base import WeightExtractor
from .gpt2 import GPT2Extractor
from .llama import LlamaExtractor
from .phi import PhiExtractor

# Map model_type from config.json to extractor class.
_REGISTRY = {
    "gpt2": GPT2Extractor,
    "llama": LlamaExtractor,
    "mistral": LlamaExtractor,
    "qwen2": LlamaExtractor,
    "gemma": LlamaExtractor,
    "gemma2": LlamaExtractor,
    "phi": PhiExtractor,
    "phi3": PhiExtractor,
    "phimoe": PhiExtractor,
}


def detect_extractor(model_path: str) -> WeightExtractor:
    """Read config.json from a HuggingFace model directory and return the extractor.

    Args:
        model_path: local path to the HF model directory (with config.json).

    Returns:
        Appropriate WeightExtractor subclass instance.

    Raises:
        ValueError: if model_type is not recognized.
        FileNotFoundError: if config.json is missing.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {config_path}. "
            f"Is {model_path} a valid HuggingFace model directory?"
        )

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "").lower()
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in {config_path}. "
            f"Supported: {sorted(_REGISTRY.keys())}"
        )

    extractor_cls = _REGISTRY[model_type]
    return extractor_cls(config)
