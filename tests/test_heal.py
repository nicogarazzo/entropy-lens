"""Tests for healing fine-tune module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from entropy_lens.heal import (
    _detect_target_modules,
    _load_dataset,
    heal_model,
)


class FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class TestDetectTargetModules:
    def test_gpt2(self):
        config = FakeConfig("gpt2")
        modules = _detect_target_modules(config)
        assert "c_attn" in modules
        assert "c_proj" in modules
        assert "c_fc" in modules

    def test_llama(self):
        config = FakeConfig("llama")
        modules = _detect_target_modules(config)
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "gate_proj" in modules

    def test_mistral(self):
        config = FakeConfig("mistral")
        modules = _detect_target_modules(config)
        assert "q_proj" in modules

    def test_phi(self):
        config = FakeConfig("phi")
        modules = _detect_target_modules(config)
        assert "fc1" in modules
        assert "fc2" in modules

    def test_unknown_falls_back_to_llama(self):
        config = FakeConfig("some_new_model")
        modules = _detect_target_modules(config)
        assert modules == ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]

    def test_case_insensitive(self):
        config = FakeConfig("GPT2")
        modules = _detect_target_modules(config)
        assert "c_attn" in modules


class TestLoadDataset:
    def test_unknown_dataset_raises(self):
        tokenizer = MagicMock()
        with pytest.raises(ValueError, match="Unknown dataset"):
            _load_dataset("nonexistent_dataset", tokenizer, 128)
