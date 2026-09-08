"""
Tests for Policy Engine and Dynamic Plan Adaptation
Verifies policy drift detection, diff calculation, and adaptation reward scoring.
"""

import unittest
from environment.aria_env import ARIAEnvironment
from environment.tools.policy_engine import PolicyEngine
from environment.reward import RewardModel


class TestPolicyAdaptation(unittest.TestCase):

    def test_policy_engine_drift(self):
        engine = PolicyEngine()
        p1 = engine.get_policy()
        self.assertEqual(engine.version, 1)
        self.assertFalse(engine.has_changed())

        # Update policy
        p2 = engine.update_policy()
        self.assertEqual(engine.version, 2)
        self.assertTrue(engine.has_changed())

        # Check diff
        diff = engine.get_diff()
        self.assertIn("changed_fields", diff)
        self.assertEqual(diff["old_version"], 1)
        self.assertEqual(diff["new_version"], 2)

    def test_environment_policy_detection(self):
        env = ARIAEnvironment(capped=False, difficulty=1)
        obs = env.reset()
        self.assertFalse(obs["policy_changed"])

        # Trigger policy change
        env.policy_engine.update_policy()
        env.state.trigger_policy_change()

        # Agent queries policy
        action = {"tool": "policy", "operation": "get", "params": {}}
        obs, reward, done, info = env.step(action)

        self.assertTrue(info.get("adaptation_detected"))
        self.assertTrue(env.state.adaptation_triggered)

    def test_reward_model_adaptation_component(self):
        rm = RewardModel(capped=False)

        # Scenario A: Policy changed and agent adapted -> full adaptation reward
        r_adapted = rm.compute(
            tasks_completed=5,
            total_tasks=5,
            tool_calls=5,
            min_tool_calls=5,
            adaptation_triggered=True,
            policy_changed=True,
            action_history=[]
        )
        breakdown_adapted = rm.get_last_reward_breakdown()
        self.assertEqual(breakdown_adapted["r3_adaptation"], 1.0)

        # Scenario B: Policy changed but agent failed to adapt -> zero adaptation reward
        r_failed = rm.compute(
            tasks_completed=5,
            total_tasks=5,
            tool_calls=5,
            min_tool_calls=5,
            adaptation_triggered=False,
            policy_changed=True,
            action_history=[]
        )
        breakdown_failed = rm.get_last_reward_breakdown()
        self.assertEqual(breakdown_failed["r3_adaptation"], 0.0)
        self.assertGreater(r_adapted, r_failed)


if __name__ == "__main__":
    unittest.main()
