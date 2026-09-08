"""
Tests for edge/device.py
Verifies truthful hardware detection and mock Snapdragon NPU validation.
"""

import unittest
from unittest.mock import patch
from edge.device import get_device_info, is_snapdragon_hardware


class TestDeviceDetection(unittest.TestCase):

    def test_device_info_structure(self):
        info = get_device_info()
        required_keys = [
            "platform",
            "architecture",
            "cpu",
            "backend",
            "accelerator",
            "qnn_available",
            "npu_available",
            "execution_providers"
        ]
        for k in required_keys:
            self.assertIn(k, info, f"Missing key in device info: {k}")

        self.assertIsInstance(info["qnn_available"], bool)
        self.assertIsInstance(info["npu_available"], bool)
        self.assertIsInstance(info["execution_providers"], list)

    def test_standard_pc_fallback(self):
        with patch("edge.device.get_onnx_execution_providers", return_value=["CPUExecutionProvider"]):
            with patch("edge.device.is_snapdragon_hardware", return_value=False):
                info = get_device_info()
                self.assertEqual(info["backend"], "cpu")
                self.assertEqual(info["accelerator"], "CPU")
                self.assertFalse(info["qnn_available"])
                self.assertFalse(info["npu_available"])

    def test_mock_snapdragon_npu_detection(self):
        with patch("edge.device.get_onnx_execution_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]):
            with patch("edge.device.is_snapdragon_hardware", return_value=True):
                info = get_device_info()
                self.assertEqual(info["backend"], "qnn")
                self.assertEqual(info["accelerator"], "NPU")
                self.assertTrue(info["qnn_available"])
                self.assertTrue(info["npu_available"])


if __name__ == "__main__":
    unittest.main()
