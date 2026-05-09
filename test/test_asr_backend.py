import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from asr_backend import (
    ASRConfig,
    SimpleHypothesis,
    TransformersASRBackend,
    resolve_asr_model,
    NEMO_MODEL_ALIASES,
    WHISPER_MODEL_ALIASES,
    TRANSFORMERS_ASR_MODEL_ALIASES,
)


class TestResolveAsrModel(unittest.TestCase):
    def test_resolve_nemo_default(self):
        result = resolve_asr_model("nemo", "")
        self.assertEqual(result, NEMO_MODEL_ALIASES["parakeet-0.6b"])

    def test_resolve_nemo_alias(self):
        result = resolve_asr_model("nemo", "parakeet-1.1b")
        self.assertEqual(result, NEMO_MODEL_ALIASES["parakeet-1.1b"])

    def test_resolve_nemo_pass_through(self):
        result = resolve_asr_model("nemo", "some/custom/model")
        self.assertEqual(result, "some/custom/model")

    def test_resolve_whisper_default(self):
        result = resolve_asr_model("whisper", "")
        self.assertEqual(result, WHISPER_MODEL_ALIASES["medical-whisper-large-v3"])

    def test_resolve_whisper_alias(self):
        result = resolve_asr_model("whisper", "medical-whisper-large-v3")
        self.assertEqual(result, WHISPER_MODEL_ALIASES["medical-whisper-large-v3"])

    def test_resolve_transformers_whisper(self):
        result = resolve_asr_model("transformers-whisper", "")
        self.assertEqual(result, WHISPER_MODEL_ALIASES["medical-whisper-large-v3"])

    def test_resolve_transformers_asr_default(self):
        result = resolve_asr_model("transformers-asr", "")
        self.assertEqual(result, TRANSFORMERS_ASR_MODEL_ALIASES["cohere-transcribe-03-2026"])

    def test_resolve_transformers_asr_alias(self):
        result = resolve_asr_model("transformers-asr", "cohere-transcribe-03-2026")
        self.assertEqual(result, "CohereLabs/cohere-transcribe-03-2026")

    def test_resolve_hf_asr_default(self):
        result = resolve_asr_model("hf-asr", "")
        self.assertEqual(result, TRANSFORMERS_ASR_MODEL_ALIASES["cohere-transcribe-03-2026"])

    def test_resolve_hf_asr_alias(self):
        result = resolve_asr_model("hf-asr", "cohere-transcribe-03-2026")
        self.assertEqual(result, "CohereLabs/cohere-transcribe-03-2026")

    def test_resolve_transformers_asr_pass_through(self):
        result = resolve_asr_model("transformers-asr", "some/custom/model")
        self.assertEqual(result, "some/custom/model")

    def test_resolve_old_transformers_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_asr_model("transformers", "")
        self.assertIn("must be", str(ctx.exception))
        self.assertIn("transformers-asr", str(ctx.exception))

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_asr_model("unknown-backend", "")
        self.assertIn("must be", str(ctx.exception))


class TestASRConfig(unittest.TestCase):
    def test_defaults(self):
        config = ASRConfig(backend="nemo", model_key="pk", model_name="m")
        self.assertFalse(config.trust_remote_code)
        self.assertFalse(config.return_timestamps)
        self.assertIsNone(config.chunk_length_s)
        self.assertIsNone(config.stride_length_s)

    def test_new_fields_passed(self):
        config = ASRConfig(
            backend="transformers-asr",
            model_key="pk",
            model_name="m",
            trust_remote_code=True,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5,
        )
        self.assertTrue(config.trust_remote_code)
        self.assertTrue(config.return_timestamps)
        self.assertEqual(config.chunk_length_s, 30)
        self.assertEqual(config.stride_length_s, 5)

    def test_stride_as_tuple(self):
        config = ASRConfig(
            backend="transformers-asr",
            model_key="pk",
            model_name="m",
            stride_length_s=(0, 2),
        )
        self.assertEqual(config.stride_length_s, (0, 2))


class TestSimpleHypothesis(unittest.TestCase):
    def test_creation(self):
        hyp = SimpleHypothesis(text="hello", timestamp={"word": [], "segment": []})
        self.assertEqual(hyp.text, "hello")
        self.assertEqual(hyp.timestamp["word"], [])
        self.assertEqual(hyp.timestamp["segment"], [])


class TestTransformersASRBackendHypothesis(unittest.TestCase):
    """Test _to_hypothesis and _prepare_input without loading a real model."""

    def setUp(self):
        self.backend = TransformersASRBackend.__new__(TransformersASRBackend)
        self.backend.return_timestamps = True

    def test_to_hypothesis_no_timestamps(self):
        hyp = self.backend._to_hypothesis({"text": "hello world"}, timestamps=False)
        self.assertEqual(hyp.text, "hello world")
        self.assertEqual(hyp.timestamp["word"], [])
        self.assertEqual(hyp.timestamp["segment"], [])

    def test_to_hypothesis_empty_text(self):
        hyp = self.backend._to_hypothesis({"text": ""}, timestamps=False)
        self.assertEqual(hyp.text, "")

    def test_to_hypothesis_none_text(self):
        hyp = self.backend._to_hypothesis({}, timestamps=False)
        self.assertEqual(hyp.text, "")

    def test_to_hypothesis_with_timestamps(self):
        hyp = self.backend._to_hypothesis({"text": "hello world"}, timestamps=True)
        self.assertEqual(hyp.text, "hello world")
        self.assertEqual(len(hyp.timestamp["word"]), 0)


class TestTransformersASRBackendPrepareInput(unittest.TestCase):
    def setUp(self):
        self.backend = TransformersASRBackend.__new__(TransformersASRBackend)

    def test_prepare_tensor_input(self):
        import numpy as np
        tensor = torch.randn(16000)
        result = self.backend._prepare_input(tensor)
        self.assertIsInstance(result, np.ndarray)

    def test_prepare_tensor_2d_input(self):
        import numpy as np
        tensor = torch.randn(1, 16000)
        result = self.backend._prepare_input(tensor)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (16000,))


class TestTransformersASRBackendTo(unittest.TestCase):
    def setUp(self):
        self.backend = TransformersASRBackend.__new__(TransformersASRBackend)

    def test_to_returns_self(self):
        result = self.backend.to("cpu")
        self.assertIs(result, self.backend)

    def test_to_with_cuda(self):
        result = self.backend.to("cuda:0")
        self.assertIs(result, self.backend)


class TestLoadASRConfig(unittest.TestCase):
    """Quick check that ASRConfig can be created in all three modes."""
    def test_nemo_config(self):
        config = ASRConfig(backend="nemo", model_key="parakeet-0.6b", model_name="nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(config.backend, "nemo")
        self.assertEqual(config.model_name, "nvidia/parakeet-tdt-0.6b-v3")

    def test_whisper_config(self):
        config = ASRConfig(
            backend="whisper",
            model_key="medical-whisper-large-v3",
            model_name="Na0s/Medical-Whisper-Large-v3",
            language="en",
            task="transcribe",
        )
        self.assertEqual(config.backend, "whisper")
        self.assertEqual(config.language, "en")

    def test_transformers_asr_config(self):
        config = ASRConfig(
            backend="transformers-asr",
            model_key="cohere-transcribe-03-2026",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            trust_remote_code=False,
            return_timestamps=False,
            chunk_length_s=None,
            stride_length_s=None,
        )
        self.assertEqual(config.backend, "transformers-asr")
        self.assertFalse(config.trust_remote_code)
        self.assertFalse(config.return_timestamps)


if __name__ == "__main__":
    unittest.main()
