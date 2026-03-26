"""FastAPI application for the Customer Support Triage OpenEnv environment."""

from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from env import CustomerSupportEnv
from models import Action, Observation, Reward
from tasks import TASKS, grade


# Initialize the FastAPI application
app = FastAPI(
    title="Customer Support Triage OpenEnv",
    description="An OpenEnv environment where an AI agent resolves customer support tickets.",
    version="1.0.0",
)

# Global environment instance
env = CustomerSupportEnv()


# Response models for FastAPI documentation
class StepResponse(BaseModel):
    """Response model for the /step endpoint."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GraderResponse(BaseModel):
    """Response model for the /grader endpoint."""

    score: float


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    """Reset the environment and return the initial observation.

    Returns:
        Observation: The initial observation for a new episode.
    """
    return env.reset()


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Apply an action to the environment.

    Args:
        action: The Action to apply (action_type and parameters).

    Returns:
        StepResponse: Contains observation, reward, done flag, and info dictionary.
    """
    try:
        observation, reward, done, info = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Get the full internal state of the environment for debugging and grading.

    Returns:
        dict: The complete unmasked environment state.
    """
    return env.state()


@app.get("/tasks")
def get_tasks() -> Dict[str, Dict[str, Any]]:
    """Get all available tasks for evaluation.

    Returns:
        dict: A dictionary of task definitions (Easy, Medium, Hard) with descriptions
              and action schemas.
    """
    return TASKS


@app.get("/grader", response_model=GraderResponse)
def get_grader(task_id: str = Query(..., description="The task ID to grade")) -> GraderResponse:
    """Grade the agent's performance on a specific task.

    Args:
        task_id: The task identifier (e.g., "task_easy", "task_medium", "task_hard").

    Returns:
        GraderResponse: A score between 0.0 and 1.0 indicating task completion level.

    Raises:
        HTTPException: If the task_id is invalid.
    """
    try:
        score = grade(env.state(), task_id)
        return GraderResponse(score=score)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
