import os
import unittest
from unittest.mock import patch

from settings import AppSettings


class TestSettings(unittest.TestCase):
    def test_api_port_defaults_to_8000(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(AppSettings.from_env().api_port, 8000)

    def test_api_port_reads_environment(self):
        with patch.dict(os.environ, {"API_PORT": "9123"}, clear=True):
            self.assertEqual(AppSettings.from_env().api_port, 9123)


if __name__ == "__main__":
    unittest.main()
