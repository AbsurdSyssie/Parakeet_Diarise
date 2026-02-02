import unittest

from chunk_transcribe import _merge_hypotheses


class FakeHyp:
    def __init__(self, words):
        self.timestamp = {"word": words}


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


if __name__ == "__main__":
    unittest.main()
