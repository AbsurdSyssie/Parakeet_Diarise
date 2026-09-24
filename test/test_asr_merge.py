import unittest

from asr_merge import (
    build_chunk_meta,
    dedup_overlaps,
    offset_words,
    parse_chunk_filename,
    reconcile_chunk_words,
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

    def test_reconcile_chunk_words_assigns_overlap_to_one_owner(self):
        chunks = [
            [
                {"word": "with", "start": 9.50, "end": 9.80},
                {"word": "business", "start": 9.75, "end": 10.15},
                {"word": "as", "start": 10.10, "end": 10.25},
            ],
            [
                {"word": "With", "start": 9.70, "end": 10.05},
                {"word": "business", "start": 9.95, "end": 10.35},
                {"word": "as", "start": 10.30, "end": 10.45},
                {"word": "usual", "start": 10.45, "end": 10.80},
            ],
        ]

        merged = reconcile_chunk_words(chunks, [(0.0, 10.5), (9.5, 20.0)])

        self.assertEqual([word["word"] for word in merged], ["with", "business", "as", "usual"])

    def test_reconcile_chunk_words_preserves_repetition_inside_owner(self):
        chunks = [
            [
                {"word": "very", "start": 8.0, "end": 8.2},
                {"word": "very", "start": 8.3, "end": 8.5},
                {"word": "hard", "start": 8.6, "end": 8.9},
            ],
            [{"word": "next", "start": 10.2, "end": 10.5}],
        ]

        merged = reconcile_chunk_words(chunks, [(0.0, 10.0), (10.0, 20.0)])

        self.assertEqual([word["word"] for word in merged], ["very", "very", "hard", "next"])

    def test_reconcile_chunk_words_removes_abutting_duplicate_at_exact_seam(self):
        chunks = [
            [{"word": "the", "start": 9.70, "end": 9.90}],
            [{"word": "the", "start": 10.02, "end": 10.20}],
        ]

        merged = reconcile_chunk_words(chunks, [(0.0, 10.0), (10.0, 20.0)])

        self.assertEqual([word["word"] for word in merged], ["the"])


if __name__ == "__main__":
    unittest.main()
