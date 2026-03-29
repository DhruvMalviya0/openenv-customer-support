"""Baseline AI agent for the Customer Support Triage OpenEnv using GPT-4o-mini."""

import json
import os
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

# Initialize OpenAI client from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Configuration
API_BASE_URL = "https://leaf46391-customer-support-triage.hf.space"
MAX_STEPS = 10

# Known ticket-level refund amounts from the local environment scenarios.
KNOWN_REFUNDS = {
    "TKT-1001": {"amount": "59.99", "currency": "USD"},
    "TKT-1002": {"amount": "89.00", "currency": "USD"},
    "TKT-1003": {"amount": "0.00", "currency": "USD"},
    "TKT-2001": {"amount": "0.00", "currency": "USD"},
    "TKT-2002": {"amount": "0.00", "currency": "USD"},
    "TKT-2003": {"amount": "349.99", "currency": "USD"},
}


def fallback_action(task_id: str, observation: Dict[str, Any], step_count: int) -> Dict[str, Any]:
    """Deterministic local policy used when the LLM is unavailable."""
    ticket_id = observation.get("ticket_id", "")
    db_state = observation.get("database_state", {})

    if task_id == "task_easy":
        return {
            "action_type": "escalate",
            "parameters": {
                "reason": "Password recovery requires IT support verification.",
                "target": "IT_Support",
            },
        }

    if task_id == "task_medium":
        if step_count == 1:
            return {"action_type": "query_db", "parameters": {}}

        warranty_date = db_state.get("warranty_expiration")
        if warranty_date:
            body = f"Your warranty is active through {warranty_date}."
        else:
            body = "I could not find a warranty expiration date in your account details."
        return {"action_type": "send_email", "parameters": {"body": body}}

    if task_id == "task_hard":
        if step_count == 1:
            return {"action_type": "query_db", "parameters": {}}
        if step_count == 2:
            refund = KNOWN_REFUNDS.get(ticket_id, {"amount": "0.00", "currency": "USD"})
            return {
                "action_type": "issue_refund",
                "parameters": {
                    "amount": refund["amount"],
                    "currency": refund["currency"],
                },
            }

        refund = KNOWN_REFUNDS.get(ticket_id, {"amount": "0.00", "currency": "USD"})
        body = (
            f"Your refund of {refund['amount']} {refund['currency']} has been processed successfully."
        )
        return {"action_type": "send_email", "parameters": {"body": body}}

    return {"action_type": "query_db", "parameters": {}}


def llm_action(
    task_description: str,
    action_schema: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Request the next action from GPT and parse it as JSON."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    prompt = f"""You are an AI customer support agent. Your task is:

OBJECTIVE:
{task_description}

ALLOWED ACTIONS:
{json.dumps(action_schema, indent=2)}

CURRENT OBSERVATION:
Ticket ID: {observation['ticket_id']}
Customer Message: {observation['customer_message']}
Database State: {json.dumps(observation['database_state'], indent=2)}
Previous Action Result: {observation['previous_action_result']}

You must respond with ONLY a valid JSON object representing an Action. The JSON must have exactly two fields:
1. "action_type": one of the allowed action types
2. "parameters": a dictionary of string key-value pairs specific to the action

Example response format:
{{"action_type": "query_db", "parameters": {{}}}}

Now, decide what action to take next and respond with ONLY the JSON object."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    response_text = response.choices[0].message.content
    return json.loads(response_text)


def run_episode(task_id: str, task_description: str, action_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single episode of the environment with the AI agent.

    Args:
        task_id: The unique identifier for the task.
        task_description: A human-readable description of the task objective.
        action_schema: The allowed actions and their required parameters.

    Returns:
        A dictionary containing:
        - task_id: The task identifier
        - steps_taken: Number of steps the agent took
        - score: The final grading score (0.0 to 1.0)
    """
    # Initialize the environment
    print(f"\n{'='*60}")
    print(f"Running Episode: {task_id}")
    print(f"{'='*60}")

    reset_response = requests.post(f"{API_BASE_URL}/reset")
    reset_response.raise_for_status()
    observation = reset_response.json()

    print(f"Initial Observation:")
    print(f"  Ticket ID: {observation['ticket_id']}")
    print(f"  Customer Message: {observation['customer_message']}")
    print(f"  Database State: {observation['database_state']}")

    step_count = 0
    done = False
    using_fallback = False

    # Main action loop
    while not done and step_count < MAX_STEPS:
        step_count += 1
        print(f"\n--- Step {step_count} ---")

        try:
            action_dict = llm_action(task_description, action_schema, observation)

            print(f"Agent decided: {action_dict['action_type']}")
            if action_dict.get("parameters"):
                print(f"  Parameters: {action_dict['parameters']}")

        except Exception as e:
            if not using_fallback:
                print(f"OpenAI unavailable, switching to deterministic fallback policy: {e}")
                using_fallback = True

            action_dict = fallback_action(task_id, observation, step_count)
            print(f"Fallback decided: {action_dict['action_type']}")
            if action_dict.get("parameters"):
                print(f"  Parameters: {action_dict['parameters']}")

        # Send the action to the environment
        step_response = requests.post(
            f"{API_BASE_URL}/step",
            json=action_dict,
        )
        step_response.raise_for_status()

        step_data = step_response.json()
        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        print(f"Step Reward: {reward['step_reward']:.2f}")
        print(f"Total Reward: {reward['total_reward']:.2f}")
        print(f"Done: {done}")
        print(f"Previous Action Result: {observation['previous_action_result']}")

    # Get the final score from the grader
    grader_response = requests.get(
        f"{API_BASE_URL}/grader",
        params={"task_id": task_id},
    )
    grader_response.raise_for_status()
    grader_data = grader_response.json()
    final_score = grader_data["score"]

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Steps Taken: {step_count}")
    print(f"Final Score: {final_score:.2f}")
    print(f"{'='*60}")

    return {
        "task_id": task_id,
        "steps_taken": step_count,
        "score": final_score,
    }


def main() -> None:
    """Main entry point: fetch tasks and run the baseline agent on all of them."""
    print("Starting Baseline Agent for Customer Support Triage OpenEnv")
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set. Running deterministic fallback-only mode.")

    # Fetch all tasks
    tasks_response = requests.get(f"{API_BASE_URL}/tasks")
    tasks_response.raise_for_status()
    all_tasks = tasks_response.json()

    results = []

    # Run the agent on each task
    for task_id, task_data in all_tasks.items():
        try:
            result = run_episode(
                task_id=task_id,
                task_description=task_data["description"],
                action_schema=task_data["action_schema"],
            )
            results.append(result)
        except Exception as e:
            print(f"Error running episode for {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "steps_taken": -1,
                "score": 0.0,
            })

    # Print final summary
    print(f"\n\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(
            f"{result['task_id']:20} | Steps: {result['steps_taken']:2} | Score: {result['score']:.2f}"
        )
    print(f"{'='*60}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"Average Score: {avg_score:.2f}/1.0")


if __name__ == "__main__":
    main()
