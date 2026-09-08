"""
ARIA Nova — FastAPI Server
Exposes OpenEnv endpoints and ARIA Nova on-device edge AI API.
Supports /api/device, /api/health, /api/task, alongside existing OpenEnv routes.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from environment.aria_env import ARIAEnvironment
from edge.device import get_device_info
from edge.health import get_health_status
from edge.agent import ARIANovaAgent

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="ARIA Nova — Enterprise AI Environment & Edge API",
    description="Adaptive On-Device Enterprise AI for Snapdragon PCs with CPU Fallback",
    version="2.0.0",
)

# Global environment instance for legacy OpenEnv compatibility
env: Optional[ARIAEnvironment] = None
global_agent = ARIANovaAgent()

# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class ActionRequest(BaseModel):
    tool: str
    operation: str
    params: Dict[str, Any] = {}

class ResetRequest(BaseModel):
    capped: bool = True
    difficulty: int = 1

class TaskRequest(BaseModel):
    task: str
    max_steps: Optional[int] = 10
    policy_drift_at: Optional[int] = 4

# ─────────────────────────────────────────────
# NEW ARIA NOVA EDGE API ROUTES
# ─────────────────────────────────────────────

@app.get("/api/device")
def api_device():
    """Return genuine device, processor, and ONNX Runtime execution provider telemetry."""
    return get_device_info()


@app.get("/api/health")
def api_health():
    """Return on-device runtime health, model load status, and offline compliance."""
    return get_health_status()


@app.post("/api/task")
def api_task(request: TaskRequest):
    """
    Run an enterprise workflow task using on-device SLM inference and adaptive planning.
    """
    result = global_agent.run_task(
        task=request.task,
        max_steps=request.max_steps or 10,
        policy_drift_at=request.policy_drift_at
    )
    return {
        "status": result["status"],
        "task": result["task"],
        "trace": result["steps"],
        "reward": result["reward"],
        "adaptation_score": result["adaptation_score"],
        "backend": result["backend"],
        "accelerator": result["accelerator"],
        "latency_ms": result.get("latency_ms", 0.0),
        "policy_changes_detected": result["policy_changes_detected"]
    }

# ─────────────────────────────────────────────
# EXISTING OPENENV ROUTES (PRESERVED)
# ─────────────────────────────────────────────

@app.get("/")
def root():
    dev = get_device_info()
    return {
        "name": "ARIA Nova",
        "tagline": "Adaptive On-Device Enterprise AI for Snapdragon PCs",
        "version": "2.0.0",
        "author": "Angel Singh",
        "backend": dev["backend"],
        "accelerator": dev["accelerator"],
        "qnn_available": dev["qnn_available"],
        "status": "running",
    }


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset environment to initial state"""
    global env
    env = ARIAEnvironment(
        capped=request.capped,
        difficulty=request.difficulty,
    )
    observation = env.reset()
    return {
        "status": "reset",
        "observation": observation,
    }


@app.post("/step")
def step(action: ActionRequest):
    """Execute one step in environment"""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    action_dict = {
        "tool": action.tool,
        "operation": action.operation,
        "params": action.params,
    }

    observation, reward, done, info = env.step(action_dict)

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Get current environment state"""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return {
        "state": env.state.summary(),
        "policy": env.policy_engine.get_policy(),
        "spreadsheet": env.spreadsheet.summary(),
    }


@app.get("/health")
def health():
    """Legacy health check"""
    return {"status": "healthy"}


@app.post("/render")
def render():
    """Render current environment state"""
    global env
    if env is None:
        return {"error": "Environment not initialized"}
    env.render()
    return {"status": "rendered"}


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)