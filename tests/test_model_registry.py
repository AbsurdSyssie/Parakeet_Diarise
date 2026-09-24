import unittest

from asr import ASRConfig
from model_registry import (
    MODEL_REGISTRY,
    default_model_key_for_backend,
    effective_timestamps,
    initial_config,
    validate_request,
)
from settings import AppSettings


def settings(**overrides):
    values = {
        "hf_token": None,
        "asr_backend": "nemo",
        "asr_model": "",
        "asr_language": "en",
        "asr_task": "transcribe",
        "asr_trust_remote_code": False,
        "asr_return_timestamps": False,
        "asr_chunk_length_s": None,
        "asr_stride_length_s": None,
        "asr_attention_implementation": None,
        "diarization_model": "nvidia/Nemotron-3-Diarization",
        "vad_sample_rate": 16000,
        "disable_cuda_graphs": False,
    }
    values.update(overrides)
    return AppSettings(**values)


class TestModelRegistry(unittest.TestCase):
    def test_default_model_keys_follow_backend(self):
        self.assertEqual(default_model_key_for_backend("nemo"), "parakeet-0.6b")
        self.assertEqual(
            default_model_key_for_backend("faster-whisper"),
            "faster-whisper-large-v3",
        )
        self.assertEqual(
            default_model_key_for_backend("transformers-asr"),
            "cohere-transcribe-03-2026",
        )

    def test_initial_config_uses_curated_default(self):
        config = initial_config(settings())
        self.assertEqual(config.backend, "nemo")
        self.assertEqual(config.model_key, "parakeet-0.6b")
        self.assertEqual(
            config.model_name,
            MODEL_REGISTRY["parakeet-0.6b"].model_name,
        )

    def test_custom_transformers_model_is_allowed(self):
        config = initial_config(
            settings(
                asr_backend="transformers-asr",
                asr_model="example/custom-asr",
                asr_trust_remote_code=True,
            )
        )
        self.assertEqual(config.model_name, "example/custom-asr")
        self.assertTrue(config.trust_remote_code)

    def test_effective_word_timestamps(self):
        config = ASRConfig(
            backend="nemo",
            model_key="parakeet-0.6b",
            model_name=MODEL_REGISTRY["parakeet-0.6b"].model_name,
            return_timestamps=True,
        )
        self.assertEqual(effective_timestamps(config, "word"), "word")

    def test_text_only_model_rejects_diarization(self):
        config = ASRConfig(
            backend="transformers-asr",
            model_key="granite-speech-4.1-2b-nar",
            model_name=MODEL_REGISTRY["granite-speech-4.1-2b-nar"].model_name,
            return_timestamps=False,
        )
        with self.assertRaises(ValueError):
            validate_request(config, diarization=True, timestamps="word")


if __name__ == "__main__":
    unittest.main()
