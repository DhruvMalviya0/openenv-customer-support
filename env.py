"""Environment logic for the Customer Support Triage OpenEnv project."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import Action, Observation, Reward


class CustomerSupportEnv:
    """Manages one customer-support ticket episode at a time."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._scenarios: List[Dict[str, Any]] = self._build_dummy_scenarios()

        self._current_ticket: Dict[str, Any] = {}
        self._visible_database_state: Dict[str, str] = {}
        self._database_revealed = False
        self._episode_done = False
        self._total_reward = 0.0
        self._step_count = 0
        self._last_action_result = ""
        self._last_email_body = ""
        self._escalation_target = ""
        self._refund_issued_amount = ""
        self._refund_issued_currency = ""

    def reset(self) -> Observation:
        """Start a new episode and return the initial observation."""
        self._current_ticket = deepcopy(self._rng.choice(self._scenarios))
        self._visible_database_state = deepcopy(self._current_ticket["initial_database_state"])
        self._database_revealed = False
        self._episode_done = False
        self._total_reward = 0.0
        self._step_count = 0
        self._last_action_result = "Ticket opened. Awaiting agent action."
        self._last_email_body = ""
        self._escalation_target = ""
        self._refund_issued_amount = ""
        self._refund_issued_currency = ""

        return Observation(
            ticket_id=self._current_ticket["ticket_id"],
            customer_message=self._current_ticket["customer_message"],
            database_state=deepcopy(self._visible_database_state),
            previous_action_result=self._last_action_result,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Apply an action and return transition data.

        Returns:
            (observation, reward, done, info)
        """
        if not self._current_ticket:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")

        if self._episode_done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        self._step_count += 1

        # Basic shaping: small step penalty + reward/penalty for valid/invalid action.
        step_reward = -0.01
        valid_action = False
        action_type = action.action_type
        params = action.parameters

        if action_type == "query_db":
            valid_action = True
            self._handle_query_db()
            self._last_action_result = "Customer database queried successfully."

        elif action_type == "issue_refund":
            valid_action = self._handle_issue_refund(params)

        elif action_type == "send_email":
            valid_action = self._handle_send_email(params)
            self._episode_done = True

        elif action_type == "escalate":
            valid_action = self._handle_escalate(params)
            self._episode_done = True

        if valid_action:
            step_reward += 0.1
        else:
            step_reward -= 0.1

        self._total_reward += step_reward

        info: Dict[str, Any] = {
            "valid_action": valid_action,
            "action_type": action_type,
            "step_count": self._step_count,
        }

        reward = Reward(
            step_reward=step_reward,
            total_reward=self._total_reward,
            done=self._episode_done,
            info=deepcopy(info),
        )

        observation = Observation(
            ticket_id=self._current_ticket["ticket_id"],
            customer_message=self._current_ticket["customer_message"],
            database_state=deepcopy(self._visible_database_state),
            previous_action_result=self._last_action_result,
        )

        return observation, reward, self._episode_done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal unmasked state for debugging/grading."""
        return {
            "current_ticket": deepcopy(self._current_ticket),
            "visible_database_state": deepcopy(self._visible_database_state),
            "database_revealed": self._database_revealed,
            "episode_done": self._episode_done,
            "total_reward": self._total_reward,
            "step_count": self._step_count,
            "last_action_result": self._last_action_result,
            "last_email_body": self._last_email_body,
            "escalation_target": self._escalation_target,
            "refund_issued_amount": self._refund_issued_amount,
            "refund_issued_currency": self._refund_issued_currency,
        }

    def _handle_query_db(self) -> None:
        """Reveal hidden customer details when the agent queries the database."""
        hidden = self._current_ticket["hidden_database_state"]
        self._visible_database_state.update(hidden)
        self._database_revealed = True

    def _handle_issue_refund(self, params: Dict[str, str]) -> bool:
        """Attempt to issue a refund if the provided parameters match the policy."""
        if not self._database_revealed:
            self._last_action_result = "Refund rejected: query_db required before issuing refunds."
            return False

        expected_amount = self._current_ticket["refund"]["amount"]
        expected_currency = self._current_ticket["refund"]["currency"]

        given_amount = params.get("amount", "")
        given_currency = params.get("currency", "")

        if given_amount == expected_amount and given_currency == expected_currency:
            self._current_ticket["order_status"] = "refunded"
            self._visible_database_state["order_status"] = "refunded"
            self._refund_issued_amount = given_amount
            self._refund_issued_currency = given_currency
            self._last_action_result = (
                f"Refund issued successfully for {expected_amount} {expected_currency}."
            )
            return True

        # Track the refund attempt even if it's wrong (for grading)
        self._refund_issued_amount = given_amount
        self._refund_issued_currency = given_currency
        self._last_action_result = (
            f"Refund rejected: expected amount={expected_amount} and currency={expected_currency}."
        )
        return False

    def _handle_send_email(self, params: Dict[str, str]) -> bool:
        """Mark the ticket resolved if an email body is supplied."""
        body = params.get("body", "").strip()
        if not body:
            self._last_action_result = "Email not sent: missing 'body' parameter."
            return False

        self._current_ticket["ticket_status"] = "resolved"
        self._last_email_body = body
        self._last_action_result = "Resolution email sent to customer. Ticket marked resolved."
        return True

    def _handle_escalate(self, params: Dict[str, str]) -> bool:
        """Escalate the ticket when a valid escalation reason is provided."""
        reason = params.get("reason", "").strip()
        target = params.get("target", "").strip()
        
        if not reason:
            self._last_action_result = "Escalation failed: missing 'reason' parameter."
            return False

        self._current_ticket["ticket_status"] = "escalated"
        self._escalation_target = target
        self._last_action_result = f"Ticket escalated to human support. Reason: {reason}"
        return True

    @staticmethod
    def _build_dummy_scenarios() -> List[Dict[str, Any]]:
        """Return a small set of hardcoded ticket scenarios for development."""
        return [
            {
                "ticket_id": "TKT-1001",
                "customer_message": (
                    "Hi, my blender arrived with a cracked jar. I'd like a refund, please."
                ),
                "ticket_status": "open",
                "order_status": "delivered_broken",
                "initial_database_state": {
                    "customer_name": "A. Rivera",
                    "order_id": "ORD-5001",
                    "order_status": "delivered_broken",
                },
                "hidden_database_state": {
                    "purchase_date": "2026-03-10",
                    "refund_eligible": "true",
                    "account_tier": "gold",
                },
                "refund": {"amount": "59.99", "currency": "USD"},
            },
            {
                "ticket_id": "TKT-1002",
                "customer_message": (
                    "I was charged twice for my headphones. Can you refund one charge?"
                ),
                "ticket_status": "open",
                "order_status": "duplicate_charge",
                "initial_database_state": {
                    "customer_name": "M. Shah",
                    "order_id": "ORD-7008",
                    "order_status": "duplicate_charge",
                },
                "hidden_database_state": {
                    "purchase_date": "2026-03-12",
                    "refund_eligible": "true",
                    "account_tier": "standard",
                },
                "refund": {"amount": "89.00", "currency": "USD"},
            },
            {
                "ticket_id": "TKT-1003",
                "customer_message": (
                    "My package never showed up and tracking hasn't updated in 10 days."
                ),
                "ticket_status": "open",
                "order_status": "in_transit_delayed",
                "initial_database_state": {
                    "customer_name": "L. Chen",
                    "order_id": "ORD-1120",
                    "order_status": "in_transit_delayed",
                },
                "hidden_database_state": {
                    "purchase_date": "2026-03-01",
                    "refund_eligible": "false",
                    "account_tier": "silver",
                },
                "refund": {"amount": "0.00", "currency": "USD"},
            },
            {
                "ticket_id": "TKT-2001",
                "customer_message": (
                    "I forgot my password and can't log in to my account. Can you help?"
                ),
                "ticket_status": "open",
                "order_status": "active",
                "initial_database_state": {
                    "customer_name": "J. Martinez",
                    "account_id": "ACC-3001",
                    "account_status": "locked",
                },
                "hidden_database_state": {
                    "last_login": "2026-03-15",
                    "account_tier": "premium",
                    "mfa_enabled": "true",
                },
                "refund": {"amount": "0.00", "currency": "USD"},
            },
            {
                "ticket_id": "TKT-2002",
                "customer_message": (
                    "Is my ThinkPad X1 laptop still under warranty? I need to know before considering repairs."
                ),
                "ticket_status": "open",
                "order_status": "delivered",
                "initial_database_state": {
                    "customer_name": "K. Wong",
                    "order_id": "ORD-8844",
                    "order_status": "delivered",
                },
                "hidden_database_state": {
                    "purchase_date": "2024-06-15",
                    "warranty_expiration": "2027-06-15",
                    "account_tier": "gold",
                },
                "refund": {"amount": "0.00", "currency": "USD"},
            },
            {
                "ticket_id": "TKT-2003",
                "customer_message": (
                    "I just received my monitor and it arrived with a permanent black line on the screen. "
                    "This is unacceptable. I want a full refund."
                ),
                "ticket_status": "open",
                "order_status": "delivered_defective",
                "initial_database_state": {
                    "customer_name": "P. Gupta",
                    "order_id": "ORD-9201",
                    "order_status": "delivered_defective",
                },
                "hidden_database_state": {
                    "purchase_date": "2026-03-20",
                    "refund_eligible": "true",
                    "account_tier": "standard",
                },
                "refund": {"amount": "349.99", "currency": "USD"},
            },
        ]
