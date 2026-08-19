import unittest

from chunk_transcribe import _merge_hypotheses, _merge_text_fragments, _merge_texts


class FakeHyp:
    def __init__(self, words, text=""):
        self.timestamp = {"word": words}
        self.text = text


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


if __name__ == "__main__":
    unittest.main()
