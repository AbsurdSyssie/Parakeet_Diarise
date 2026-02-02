import unittest

from diarize_align import assign_speakers, group_words_into_segments


class TestDiarizeAlign(unittest.TestCase):
    def test_assign_speakers_overlap(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
        ]
        turns = [
            {"start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
            {"start": 0.6, "end": 1.0, "speaker": "SPEAKER_01"},
        ]
        assigned = assign_speakers(words, turns, min_overlap_s=0.01)
        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertEqual(assigned[1]["speaker"], "SPEAKER_01")

    def test_assign_speakers_unknown(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.02}]
        turns = [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        assigned = assign_speakers(words, turns, min_overlap_s=0.03)
        self.assertEqual(assigned[0]["speaker"], "UNKNOWN")

    def test_group_words(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.2, "speaker": "S0"},
            {"word": "b", "start": 0.25, "end": 0.4, "speaker": "S0"},
            {"word": "c", "start": 1.2, "end": 1.4, "speaker": "S1"},
        ]
        segments = group_words_into_segments(words, max_gap_s=0.6)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "a b")
        self.assertEqual(segments[1]["speaker"], "S1")


if __name__ == "__main__":
    unittest.main()
