"""
Tests for server.py FastAPI routes
Verifies /api/device, /api/health, /api/task, and legacy OpenEnv endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from server import app


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("name", data)
        self.assertEqual(data["name"], "ARIA Nova")
        self.assertIn("backend", data)

    def test_api_device_endpoint(self):
        resp = self.client.get("/api/device")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("platform", data)
        self.assertIn("architecture", data)
        self.assertIn("backend", data)
        self.assertIn("accelerator", data)
        self.assertIn("qnn_available", data)

    def test_api_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_loaded", data)
        self.assertIn("backend", data)
        self.assertIn("offline_mode", data)

    def test_api_task_endpoint(self):
        payload = {
            "task": "Arrange international briefing for sales lead.",
            "max_steps": 4,
            "policy_drift_at": 2
        }
        resp = self.client.post("/api/task", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("trace", data)
        self.assertIn("reward", data)
        self.assertIn("adaptation_score", data)
        self.assertIn("backend", data)

    def test_legacy_reset_and_step(self):
        # Reset
        reset_resp = self.client.post("/reset", json={"capped": True, "difficulty": 1})
        self.assertEqual(reset_resp.status_code, 200)
        self.assertEqual(reset_resp.json()["status"], "reset")

        # Step
        step_resp = self.client.post("/step", json={
            "tool": "email",
            "operation": "list",
            "params": {}
        })
        self.assertEqual(step_resp.status_code, 200)
        step_data = step_resp.json()
        self.assertIn("observation", step_data)
        self.assertIn("reward", step_data)


if __name__ == "__main__":
    unittest.main()
