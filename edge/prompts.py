"""
ARIA Nova Prompts and Action Parsers
Builds structured contextual prompts for on-device SLMs and parses structured actions.
"""

from typing import Dict, Any


SYSTEM_PROMPT = """You are ARIA Nova, an adaptive on-device enterprise AI agent for Snapdragon PCs.
You complete complex enterprise workflows using available workplace tools while adhering to enterprise policy.
If the policy engine updates policies during execution, you must inspect the change and dynamically adapt your plan.
"""


def build_agent_prompt(task: str, observation: Dict[str, Any], current_policy: Dict[str, Any]) -> str:
    """Format environment observation and task into an SLM prompt."""
    policy_changed = observation.get("policy_changed", False)
    return f"""{SYSTEM_PROMPT}

TASK:
{task}

CURRENT ENTERPRISE POLICY:
{current_policy}

WORKFLOW STATUS:
- Step: {observation.get('step', 0)}/{observation.get('max_steps', 20)}
- Tasks Completed: {observation.get('tasks_completed', 0)}/{observation.get('total_tasks', 5)}
- Policy Changed Mid-Task: {policy_changed}
- Adaptation Triggered: {observation.get('adaptation_triggered', False)}

TOOLS AVAILABLE:
- email: list, read, send (to, subject, body)
- calendar: check, schedule, reschedule (slot, event)
- document: list, read (doc_name)
- spreadsheet: read, write (field, value)
- policy: get

INSTRUCTIONS:
1. Review current policy rules and task requirements.
2. If policy changed, inspect the new policy immediately or adapt the approval chain.
3. Output the single next action in this exact format:
TOOL: <tool_name>
OPERATION: <operation_name>
PARAMS: <key1=value1, key2=value2>

Your action:"""


def parse_action_response(response: str) -> Dict[str, Any]:
    """Parse text generation into structured OpenEnv action dictionary."""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    tool = "policy"
    operation = "get"
    params = {}

    for line in lines:
        upper = line.upper()
        if upper.startswith("TOOL:"):
            tool = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("OPERATION:"):
            operation = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("PARAMS:"):
            raw_params = line.split(":", 1)[1].strip()
            if raw_params and raw_params.lower() != "none" and raw_params != "{}":
                for pair in raw_params.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k.strip()] = v.strip()

    return {
        "tool": tool,
        "operation": operation,
        "params": params,
    }
