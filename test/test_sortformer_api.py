import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import api


class FakeSortformer:
    def __init__(self):
        self.kwargs = None

    def diarize(self, **kwargs):
        self.kwargs = kwargs
        segments = [["0.00 0.40 speaker_0"]]
        probabilities = [torch.ones(20, 4)]  # includes padded frames
        return segments, probabilities


class TestSortformerApi(unittest.TestCase):
    def test_run_requests_soft_outputs_postprocessing_and_trims_padding(self):
        model = FakeSortformer()
        waveform = torch.zeros(16_000)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "api._get_diar_model", return_value=model
        ), patch("api.torchaudio.save"):
            turns, probabilities, frame_duration_s = api._run_sortformer(
                waveform, 16_000, Path(tmpdir)
            )

        self.assertTrue(model.kwargs["include_tensor_outputs"])
        self.assertTrue(model.kwargs["postprocessing_yaml"].endswith("configs/sortformer_postprocessing.yaml"))
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(probabilities.shape, (13, 4))
        self.assertEqual(frame_duration_s, 0.08)


if __name__ == "__main__":
    unittest.main()
