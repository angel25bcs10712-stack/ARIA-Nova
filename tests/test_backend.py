"""
Tests for edge/inference.py, edge/cpu_backend.py, and edge/qnn_backend.py
Verifies CPUBackend and QNNBackend fallback behaviors.
"""

import unittest
from unittest.mock import patch
from edge.inference import get_backend
from edge.cpu_backend import CPUBackend
from edge.qnn_backend import QNNBackend


class TestBackends(unittest.TestCase):

    def test_cpu_backend(self):
        backend = CPUBackend()
        self.assertTrue(backend.is_available())
        self.assertEqual(backend.name, "cpu")
        self.assertEqual(backend.accelerator, "CPU")

        # Test model loading with non-existent path (deterministic fallback engine)
        loaded = backend.load_model("./models/non_existent")
        self.assertTrue(loaded)
        self.assertTrue(backend.is_loaded)

        # Test generation
        resp = backend.generate("TASK: Check calendar\nACTION:")
        self.assertIsInstance(resp, str)
        self.assertIn("TOOL:", resp)

        telemetry = backend.get_telemetry()
        self.assertEqual(telemetry["backend"], "cpu")
        self.assertEqual(telemetry["active_provider"], "CPUExecutionProvider")

    def test_qnn_backend_fallback_when_unavailable(self):
        with patch("edge.qnn_backend.get_onnx_execution_providers", return_value=["CPUExecutionProvider"]):
            qnn = QNNBackend()
            self.assertFalse(qnn.is_available())
            self.assertTrue(qnn.fallback_occurred)
            self.assertEqual(qnn.accelerator, "CPU")
            self.assertIn("not available", qnn.fallback_reason.lower())
            self.assertEqual(qnn.active_provider, "CPUExecutionProvider")

            # Generation works via fallback without throwing
            out = qnn.generate("TASK: Check emails\nACTION:")
            self.assertIsInstance(out, str)

    def test_backend_factory(self):
        cpu = get_backend("cpu")
        self.assertIsInstance(cpu, CPUBackend)

        qnn = get_backend("qnn")
        self.assertIsInstance(qnn, QNNBackend)


if __name__ == "__main__":
    unittest.main()
