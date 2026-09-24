import unittest
import tempfile
from pathlib import Path

from chunk_transcribe import (
    _merge_hypotheses,
    _merge_text_fragments,
    _merge_texts,
    transcribe_chunks_in_memory,
    transcribe_chunks_with_model,
)


class FakeHyp:
    def __init__(self, words, text=""):
        self.timestamp = {"word": words}
        self.text = text


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def transcribe(self, _waveforms, **_kwargs):
        return self.outputs


class TestChunkTranscribe(unittest.TestCase):
    def test_merge_hypotheses_offsets(self):
        outputs = [
            FakeHyp([
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9},
            ]),
            FakeHyp([
                {"word": "again", "start": 0.0, "end": 0.3},
            ]),
        ]
        merged = _merge_hypotheses(outputs, [1.0, 5.0])
        self.assertEqual(merged[0]["start"], 1.0)
        self.assertEqual(merged[1]["end"], 1.9)
        self.assertEqual(merged[2]["start"], 5.0)

    def test_merge_texts_removes_chunk_overlap(self):
        outputs = [
            FakeHyp([], "The patient has chest pain and nausea."),
            FakeHyp([], "chest pain and nausea. Blood pressure is normal."),
        ]
        merged = _merge_texts(outputs, [0.0, 10.0])
        self.assertEqual(
            merged,
            "The patient has chest pain and nausea. Blood pressure is normal.",
        )

    def test_merge_texts_removes_trailing_language_tags(self):
        outputs = [
            FakeHyp([], "This is my voice now you speak. <en-US>"),
            FakeHyp([], "A second sentence. <en-US>"),
        ]

        merged = _merge_texts(outputs, [0.0, 10.0])

        self.assertEqual(merged, "This is my voice now you speak. A second sentence.")

    def test_merge_texts_preserves_non_language_angle_brackets(self):
        merged = _merge_texts([FakeHyp([], "Use <example> here.")], [0.0])

        self.assertEqual(merged, "Use <example> here.")

    def test_merge_text_fragments_removes_overlap_between_batches(self):
        merged = _merge_text_fragments(
            [
                "one two three four",
                "three four five six",
                "five six seven",
            ]
        )
        self.assertEqual(merged, "one two three four five six seven")

    def test_memory_transcription_reconciles_chunk_seam_before_flattening(self):
        model = FakeModel(
            [
                FakeHyp(
                    [
                        {"word": "with", "start": 9.50, "end": 9.80},
                        {"word": "business", "start": 9.75, "end": 10.15},
                    ]
                ),
                FakeHyp(
                    [
                        {"word": "With", "start": 0.20, "end": 0.55},
                        {"word": "business", "start": 0.45, "end": 0.85},
                        {"word": "as", "start": 0.80, "end": 0.95},
                        {"word": "usual", "start": 0.95, "end": 1.30},
                    ]
                ),
            ]
        )
        chunks = [
            {"waveform": object(), "start_s": 0.0, "end_s": 10.5},
            {"waveform": object(), "start_s": 9.5, "end_s": 20.0},
        ]

        words = transcribe_chunks_in_memory(model, chunks, batch_size=2)

        self.assertEqual([word["word"] for word in words], ["with", "business", "as", "usual"])

    def test_file_transcription_uses_the_same_seam_reconciliation(self):
        model = FakeModel(
            [
                FakeHyp([{"word": "business", "start": 9.75, "end": 10.15}]),
                FakeHyp(
                    [
                        {"word": "business", "start": 0.45, "end": 0.85},
                        {"word": "usual", "start": 0.95, "end": 1.30},
                    ]
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_dir = Path(tmpdir)
            (chunk_dir / "chunk_0001_0.000-10.500.wav").touch()
            (chunk_dir / "chunk_0002_9.500-20.000.wav").touch()
            words = transcribe_chunks_with_model(
                model,
                chunk_dir,
                pad_left_s=0.0,
                pad_right_s=0.0,
                batch_size=2,
            )

        self.assertEqual([word["word"] for word in words], ["business", "usual"])


if __name__ == "__main__":
    unittest.main()
