import unittest

import torch

from diarize_align import assign_speakers, assign_speakers_from_probabilities, group_words_into_segments


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

    def test_probability_alignment_scores_partial_frames(self):
        words = [{"word": "boundary", "start": 0.04, "end": 0.12}]
        probabilities = torch.tensor([[0.9, 0.1], [0.3, 0.7]])

        assigned = assign_speakers_from_probabilities(
            words,
            probabilities,
            audio_duration_s=0.16,
            frame_duration_s=0.08,
            confidence_threshold=0.0,
            margin_threshold=0.0,
        )

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertAlmostEqual(assigned[0]["speaker_confidence"], 0.6)
        self.assertAlmostEqual(assigned[0]["speaker_margin"], 0.2)

    def test_probability_alignment_preserves_overlap_winner_and_margin(self):
        words = [{"word": "yes", "start": 0.0, "end": 0.08}]
        probabilities = torch.tensor([[0.82, 0.70, 0.02, 0.01]])

        assigned = assign_speakers_from_probabilities(
            words, probabilities, audio_duration_s=0.08, frame_duration_s=0.08
        )

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertAlmostEqual(assigned[0]["speaker_margin"], 0.12)

    def test_probability_alignment_keeps_tie_unknown(self):
        assigned = assign_speakers_from_probabilities(
            [{"word": "maybe", "start": 0.0, "end": 0.08}],
            [[0.6, 0.6]],
            audio_duration_s=0.08,
            frame_duration_s=0.08,
        )
        self.assertEqual(assigned[0]["speaker"], "UNKNOWN")

    def test_probability_alignment_prunes_to_globally_active_channels(self):
        probabilities = torch.tensor(
            [[0.05, 0.10, 0.85, 0.70], [0.05, 0.10, 0.80, 0.75]]
        )
        assigned = assign_speakers_from_probabilities(
            [{"word": "active", "start": 0.0, "end": 0.08}],
            probabilities,
            audio_duration_s=0.16,
            frame_duration_s=0.08,
            num_speakers=2,
        )
        self.assertEqual(assigned[0]["speaker"], "SPEAKER_02")

    def test_probability_alignment_smooths_only_low_confidence_word(self):
        words = [
            {"word": "one", "start": 0.0, "end": 0.08},
            {"word": "two", "start": 0.08, "end": 0.16},
            {"word": "three", "start": 0.16, "end": 0.24},
        ]
        assigned = assign_speakers_from_probabilities(
            words,
            [[0.9, 0.1], [0.30, 0.28], [0.85, 0.1]],
            audio_duration_s=0.24,
            frame_duration_s=0.08,
        )
        self.assertEqual([word["speaker"] for word in assigned], ["SPEAKER_00"] * 3)

    def test_probability_alignment_leaves_very_low_confidence_unknown(self):
        assigned = assign_speakers_from_probabilities(
            [{"word": "noise", "start": 0.0, "end": 0.08}],
            [[0.05, 0.04]],
            audio_duration_s=0.08,
            frame_duration_s=0.08,
        )
        self.assertEqual(assigned[0]["speaker"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
