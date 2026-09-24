import unittest

from diarization import normalize_speaker, parse_diarization_output


class TestDiarization(unittest.TestCase):
    def test_normalize_speaker(self):
        self.assertEqual(normalize_speaker(2), "SPEAKER_02")
        self.assertEqual(normalize_speaker("3"), "SPEAKER_03")
        self.assertEqual(normalize_speaker("speaker_4"), "SPEAKER_04")
        self.assertEqual(normalize_speaker("SPEAKER_05"), "SPEAKER_05")
        self.assertEqual(normalize_speaker(None), "UNKNOWN")

    def test_parse_mixed_segment_formats(self):
        raw = [[
            {"start": 3.0, "end": 4.0, "speaker": 1},
            "0.0 1.0 speaker_0",
            (1.5, 2.5, "2"),
            {"start_time": 4.5, "end_time": 5.0, "speaker_label": "SPEAKER_03"},
        ]]

        self.assertEqual(
            parse_diarization_output(raw),
            [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_02"},
                {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
                {"start": 4.5, "end": 5.0, "speaker": "SPEAKER_03"},
            ],
        )

    def test_parse_skips_malformed_segments(self):
        raw = [[
            "not-a-segment",
            ("bad", 1.0, "speaker_0"),
            object(),
        ]]
        self.assertEqual(parse_diarization_output(raw), [])


if __name__ == "__main__":
    unittest.main()
