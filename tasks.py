"""Task definitions and grading logic for the Customer Support Triage environment."""

from typing import Any, Dict

from models import Action


# Task definitions
TASKS: Dict[str, Dict[str, Any]] = {
    "task_easy": {
        "id": "task_easy",
        "name": "Password Recovery Request",
        "description": (
            "A customer forgot their password and needs help recovering their account. "
            "The agent must escalate the ticket to IT_Support to handle the account recovery."
        ),
        "action_schema": {
            "required_action_type": "escalate",
            "required_parameters": ["reason"],
            "escalation_target": "IT_Support",
        },
    },
    "task_medium": {
        "id": "task_medium",
        "name": "Warranty Status Inquiry",
        "description": (
            "A customer wants to know if their laptop is still under warranty. "
            "The agent must query the database to find the warranty expiration date, "
            "then send an email with the exact warranty information."
        ),
        "action_schema": {
            "required_actions": ["query_db", "send_email"],
            "required_parameters": ["body"],  # body must contain warranty date
        },
    },
    "task_hard": {
        "id": "task_hard",
        "name": "Broken Monitor Refund Request",
        "description": (
            "A customer received a broken monitor and wants a refund. "
            "The agent must query the database for the order total, "
            "issue a refund for the exact amount, and send an email confirming the refund."
        ),
        "action_schema": {
            "required_actions": ["query_db", "issue_refund", "send_email"],
            "required_parameters": ["amount", "currency", "body"],
        },
    },
}


def grade(state: Dict[str, Any], task_id: str) -> float:
    """Evaluate the agent's performance on a specific task.

    Args:
        state: The environment's internal state (from env.state())
        task_id: The task identifier (e.g., "task_easy", "task_medium", "task_hard")

    Returns:
        A score between 0.0 and 1.0:
        - 1.0: Perfect task completion
        - 0.5-0.8: Partial completion
        - 0.0: Task not completed or failed
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    if task_id == "task_easy":
        return _grade_easy(state)
    elif task_id == "task_medium":
        return _grade_medium(state)
    elif task_id == "task_hard":
        return _grade_hard(state)

    return 0.0


def _grade_easy(state: Dict[str, Any]) -> float:
    """Grade the password recovery task.

    Returns:
        1.0 if ticket was escalated to IT_Support
        0.0 otherwise
    """
    ticket = state.get("current_ticket", {})
    escalation_target = state.get("escalation_target", "")

    if ticket.get("ticket_status") == "escalated" and escalation_target == "IT_Support":
        return 1.0

    return 0.0


def _grade_medium(state: Dict[str, Any]) -> float:
    """Grade the warranty inquiry task.

    Returns:
        1.0 if email was sent containing the correct warranty expiration date
        0.5 if email was sent but the warranty date is missing or incorrect
        0.0 if no email was sent or task not attempted
    """
    ticket = state.get("current_ticket", {})
    last_email_body = state.get("last_email_body", "")
    visible_db = state.get("visible_database_state", {})

    # Email must have been sent
    if ticket.get("ticket_status") != "resolved" or not last_email_body:
        return 0.0

    # Check if the warranty expiration date from the database is in the email
    warranty_expiration = visible_db.get("warranty_expiration", "")

    if warranty_expiration and warranty_expiration in last_email_body:
        return 1.0

    # Email was sent but missing or incorrect warranty date
    if last_email_body:
        return 0.5

    return 0.0


def _grade_hard(state: Dict[str, Any]) -> float:
    """Grade the broken monitor refund task.

    Returns:
        1.0 if refund issued for exact amount AND email was sent
        0.8 if refund is correct but no email was sent
        0.5 if refund was issued but for the wrong amount
        0.0 if no refund issued or task not attempted
    """
    ticket = state.get("current_ticket", {})
    refund_issued_amount = state.get("refund_issued_amount", "")
    refund_issued_currency = state.get("refund_issued_currency", "")
    last_email_body = state.get("last_email_body", "")

    # Expected refund from the ticket scenario
    expected_amount = ticket.get("refund", {}).get("amount", "")
    expected_currency = ticket.get("refund", {}).get("currency", "")

    # No refund issued at all
    if not refund_issued_amount:
        return 0.0

    # Refund issued for wrong amount
    if refund_issued_amount != expected_amount or refund_issued_currency != expected_currency:
        return 0.5

    # Refund is correct but no email confirmation sent
    if not last_email_body or ticket.get("ticket_status") != "resolved":
        return 0.8

    # Perfect: correct refund AND email sent
    return 1.0
