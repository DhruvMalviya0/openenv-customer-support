"""Data models for the Customer Support Triage OpenEnv environment."""

from typing import Any, Dict, Literal

from pydantic import BaseModel


class Observation(BaseModel):
    """Represents what the agent observes at each environment step."""

    ticket_id: str
    customer_message: str
    database_state: Dict[str, str]
    previous_action_result: str


class Action(BaseModel):
    """Represents the agent action for the current environment step."""

    action_type: Literal["query_db", "issue_refund", "send_email", "escalate"]
    parameters: Dict[str, str]


class Reward(BaseModel):
    """Represents reward feedback after an action is processed."""

    step_reward: float
    total_reward: float
    done: bool
    info: Dict[str, Any]
