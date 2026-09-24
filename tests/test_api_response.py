import unittest

from api_response import build_response


class TestAPIResponse(unittest.TestCase):
    def test_build_response(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
        ]
        segments = [
            {
                "id": 0,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 0.9,
                "text": "hello world",
            }
        ]
        resp = build_response(words, segments, language="en", duration=1.0)
        self.assertEqual(resp["text"], "hello world")
        self.assertEqual(resp["language"], "en")
        self.assertEqual(resp["duration"], 1.0)
        self.assertEqual(resp["speakers"], ["SPEAKER_00"])

    def test_build_response_text_override(self):
        resp = build_response([], [], language="en", duration=2.5, text="override")
        self.assertEqual(resp["text"], "override")
        self.assertEqual(resp["speakers"], [])


if __name__ == "__main__":
    unittest.main()
