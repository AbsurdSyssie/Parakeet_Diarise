import unittest

import asr
import asr_backend


class TestASRPackageCompatibility(unittest.TestCase):
    def test_compatibility_facade_exports_shared_objects(self):
        self.assertIs(asr_backend.ASRConfig, asr.ASRConfig)
        self.assertIs(asr_backend.SimpleHypothesis, asr.SimpleHypothesis)
        self.assertIs(asr_backend.resolve_asr_model, asr.resolve_asr_model)
        self.assertIs(asr_backend.load_asr_backend, asr.load_asr_backend)


if __name__ == "__main__":
    unittest.main()
