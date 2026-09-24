import unittest

from asr_merge import (
    build_chunk_meta,
    dedup_overlaps,
    offset_words,
    parse_chunk_filename,
    sort_words,
)


class TestASRMergeHelpers(unittest.TestCase):
    def test_parse_chunk_filename(self):
        chunk_id, start_s, end_s = parse_chunk_filename("chunk_0026_134.01-145.67.wav")
        self.assertEqual(chunk_id, 26)
        self.assertAlmostEqual(start_s, 134.01)
        self.assertAlmostEqual(end_s, 145.67)

    def test_build_chunk_meta(self):
        meta = build_chunk_meta("chunk_0001_0.28-3.75.wav", pad_left_s=0.2, pad_right_s=0.2)
        self.assertAlmostEqual(meta.orig_start_s, 0.28)
        self.assertAlmostEqual(meta.effective_start_s, 0.08)
        self.assertAlmostEqual(meta.effective_end_s, 3.95)

    def test_offset_and_dedup(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        shifted = offset_words(words, 10.0)
        self.assertAlmostEqual(shifted[0]["start"], 10.0)
        self.assertAlmostEqual(shifted[1]["end"], 11.0)

        overlap_words = [
            {"word": "a", "start": 0.0, "end": 0.4},
            {"word": "a", "start": 0.1, "end": 0.35},
            {"word": "b", "start": 0.45, "end": 0.6},
        ]
        deduped = dedup_overlaps(sort_words(overlap_words))
        self.assertEqual([w["word"] for w in deduped], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
