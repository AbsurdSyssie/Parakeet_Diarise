import json
import unittest
from pathlib import Path

from diarize_align import assign_speakers, group_words_into_segments


class TestMergeDiarized(unittest.TestCase):
    def test_merge_pipeline(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
        ]
        turns = [
            {"start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
            {"start": 0.6, "end": 1.0, "speaker": "SPEAKER_01"},
        ]
        assigned = assign_speakers(words, turns, min_overlap_s=0.01)
        segments = group_words_into_segments(assigned, max_gap_s=0.6)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")
        self.assertEqual(segments[1]["speaker"], "SPEAKER_01")

    def test_no_turns(self):
        words = [{"word": "hello", "start": 0.0, "end": 0.4}]
        assigned = assign_speakers(words, [], min_overlap_s=0.01)
        self.assertEqual(assigned[0]["speaker"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
