import unittest
from unittest.mock import patch

import torch

from vad_chunk import run_vad_chunks_in_memory_from_waveform


class TestVadChunk(unittest.TestCase):
    def _run(self, timestamps, **overrides):
        calls = []

        def get_timestamps(_audio, _model, **kwargs):
            calls.append(kwargs)
            return timestamps

        params = dict(
            waveform=torch.zeros(16_000 * 20),
            sample_rate=16_000,
            threshold=0.3,
            min_speech_ms=150,
            min_silence_ms=220,
            merge_gap_ms=200,
            target_min_s=10.0,
            target_max_s=20.0,
            hard_max_s=30.0,
            overlap_s=1.0,
            speech_pad_ms=80,
            asr_context_pad_ms=250,
            target_merge_max_gap_s=1.5,
            energy_gate_override=False,
        )
        params.update(overrides)
        with patch("vad_chunk._load_vad", return_value=(object(), (get_timestamps, None, None, None, None))):
            chunks = run_vad_chunks_in_memory_from_waveform(**params)
        return chunks, calls

    def test_silero_gets_vad_pad_and_long_speech_limit(self):
        _, calls = self._run([{"start": 1600, "end": 3200}])
        self.assertEqual(calls[0]["speech_pad_ms"], 80)
        self.assertEqual(calls[0]["max_speech_duration_s"], 30.0)

    def test_asr_context_pad_is_separate_from_vad_pad(self):
        chunks, _ = self._run([{"start": 16_000, "end": 32_000}])
        self.assertEqual(chunks[0]["start_s"], 1.0)
        self.assertEqual(chunks[0]["waveform"].numel(), 24_000)

    def test_target_aggregation_does_not_cross_large_silence(self):
        chunks, _ = self._run(
            [{"start": 0, "end": 16_000}, {"start": 16_000 * 10, "end": 16_000 * 11}]
        )
        self.assertEqual(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()
