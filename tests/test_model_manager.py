"""
Tests for edge/model_manager.py
Verifies model presence checks, initialization, and status telemetry.
"""

import unittest
from edge.model_manager import ModelManager


class TestModelManager(unittest.TestCase):

    def test_missing_model_handling(self):
        manager = ModelManager(model_path="./models/does_not_exist_model")
        self.assertFalse(manager.is_model_present())

        # Should still initialize gracefully with deterministic local engine
        ready = manager.initialize("cpu")
        self.assertTrue(ready)
        self.assertTrue(manager.is_ready)

        status = manager.get_status()
        self.assertFalse(status["model_present"])
        self.assertTrue(status["model_ready"])
        self.assertEqual(status["backend"], "cpu")

    def test_generation_via_manager(self):
        manager = ModelManager()
        manager.initialize("cpu")
        out = manager.generate("TASK: Review budget\nACTION:")
        self.assertIsInstance(out, str)
        self.assertIn("TOOL:", out)


if __name__ == "__main__":
    unittest.main()
