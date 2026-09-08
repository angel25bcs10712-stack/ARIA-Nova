"""
Tests for edge/agent.py
Verifies end-to-end task execution, trace structure, and reward computation.
"""

import unittest
from edge.agent import ARIANovaAgent


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ARIANovaAgent(backend_type="cpu")

    def test_run_task_structure(self):
        result = self.agent.run_task(
            task="Arrange an international business meeting for an employee according to company policy.",
            max_steps=5,
            policy_drift_at=3
        )

        expected_keys = [
            "task",
            "status",
            "backend",
            "accelerator",
            "latency_ms",
            "steps",
            "policy_changes_detected",
            "adaptation_score",
            "reward",
            "tasks_completed",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

        self.assertEqual(result["status"], "completed")
        self.assertIsInstance(result["steps"], list)
        self.assertGreater(len(result["steps"]), 0)

        # Check step format
        step0 = result["steps"][0]
        self.assertIn("tool", step0)
        self.assertIn("operation", step0)
        self.assertIn("reward", step0)
        self.assertIn("latency_ms", step0)

        # Check reward and adaptation score types
        self.assertIsInstance(result["reward"], float)
        self.assertIsInstance(result["adaptation_score"], float)


if __name__ == "__main__":
    unittest.main()
