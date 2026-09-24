import unittest

from pathlib import Path


class TestHealthEndpoints(unittest.TestCase):
    def test_api_health_route(self):
        text = Path("api.py").read_text()
        self.assertIn('@app.get("/health")', text)

    def test_diarize_health_route(self):
        text = Path("diarize_api.py").read_text()
        self.assertIn('@app.get("/health")', text)
        self.assertIn('"diarization_loaded": is_diarizer_loaded()', text)


if __name__ == "__main__":
    unittest.main()
