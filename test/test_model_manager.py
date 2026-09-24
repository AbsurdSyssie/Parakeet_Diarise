import unittest
from unittest.mock import Mock, patch

from asr import ASRConfig
from model_manager import (
    ASRModelManager,
    ModelSwitchError,
    ModelUnavailableError,
)
from settings import AppSettings


def settings():
    return AppSettings(
        hf_token=None,
        asr_backend="nemo",
        asr_model="",
        asr_language="en",
        asr_task="transcribe",
        asr_trust_remote_code=False,
        asr_return_timestamps=False,
        asr_chunk_length_s=None,
        asr_stride_length_s=None,
        asr_attention_implementation=None,
        vad_sample_rate=16000,
        disable_cuda_graphs=False,
    )


def config(key):
    return ASRConfig(
        backend="nemo",
        model_key=key,
        model_name=f"example/{key}",
    )


class TestASRModelManager(unittest.TestCase):
    def test_require_active_rejects_empty_manager(self):
        manager = ASRModelManager(settings())
        with self.assertRaises(ModelUnavailableError):
            manager.require_active()

    def test_failed_switch_restores_previous_model(self):
        manager = ASRModelManager(settings())
        previous = config("previous")
        requested = config("requested")
        previous_model = object()

        manager.model = object()
        manager.active_config = previous
        manager._load_model = Mock(
            side_effect=[RuntimeError("new model failed"), previous_model]
        )

        with patch("model_manager.unload_diarizer"):
            with self.assertRaises(ModelSwitchError) as ctx:
                manager.switch(requested)

        self.assertIn("previous model was restored", str(ctx.exception.detail))
        self.assertIs(manager.model, previous_model)
        self.assertEqual(manager.active_config, previous)
        self.assertEqual(manager.state["state"], "ready")

    def test_failed_restore_leaves_no_active_model(self):
        manager = ASRModelManager(settings())
        previous = config("previous")
        requested = config("requested")

        manager.model = object()
        manager.active_config = previous
        manager._load_model = Mock(
            side_effect=[
                RuntimeError("new model failed"),
                RuntimeError("restore failed"),
            ]
        )

        with patch("model_manager.unload_diarizer"):
            with self.assertRaises(ModelSwitchError):
                manager.switch(requested)

        self.assertIsNone(manager.model)
        self.assertIsNone(manager.active_config)
        self.assertEqual(manager.state["state"], "error")


if __name__ == "__main__":
    unittest.main()
