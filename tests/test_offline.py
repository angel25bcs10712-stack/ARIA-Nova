"""
Tests for edge/offline.py
Verifies offline mode enforcement, network violation trapping, and privacy status.
"""

import unittest
from unittest.mock import patch
from edge.offline import is_offline_mode, assert_offline_compliant, get_privacy_status, OfflineViolationError


class TestOfflineMode(unittest.TestCase):

    def test_offline_status(self):
        status = get_privacy_status()
        self.assertIn("offline_mode", status)
        self.assertIn("cloud_llm_calls_blocked", status)
        self.assertIn("data_residency", status)

    def test_offline_guard_blocks_cloud(self):
        with patch("edge.offline.is_offline_mode", return_value=True):
            with self.assertRaises(OfflineViolationError):
                assert_offline_compliant("api.openai.com")

    def test_offline_guard_allows_when_disabled(self):
        with patch("edge.offline.is_offline_mode", return_value=False):
            # Should not raise
            try:
                assert_offline_compliant("api.openai.com")
            except OfflineViolationError:
                self.fail("assert_offline_compliant raised OfflineViolationError unexpectedly when offline=False")


if __name__ == "__main__":
    unittest.main()
