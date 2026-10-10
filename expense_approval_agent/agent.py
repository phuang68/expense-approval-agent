# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ambient expense-approval agent built as an ADK 2.0 graph workflow.

An expense report arrives as a JSON event (Pub/Sub push or local test payload).
- Expenses under $100 are automatically approved in Python code without calling an LLM.
- Expenses of $100 or more are analyzed by an LLM for risk factors, an alert is raised,
  and the workflow pauses for human approval via RequestInput (HITL). Once decided,
  the outcome is recorded.
"""

import base64
import json
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events import Event, RequestInput
from google.adk.workflow import Workflow
from pydantic import BaseModel, Field

from .config import config

# ---------------------------------------------------------------------------
# Pydantic schema for structured expense data
# ---------------------------------------------------------------------------


class ExpenseData(BaseModel):
    """Normalized expense details extracted from the incoming report."""

    amount: float = Field(description="Expense amount in USD")
    submitter: str = Field(description="Name or email of the submitter")
    category: str = Field(description="Expense category (e.g., travel, meals, equipment)")
    description: str = Field(description="Business justification / description")
    date: str = Field(description="Date of the expense (YYYY-MM-DD)")


# ---------------------------------------------------------------------------
# Node 1: Ingestion & normalization (Pub/Sub base64 or plain JSON)
# ---------------------------------------------------------------------------


def parse_expense_event(node_input: Any) -> Event:
    """Parse an incoming event payload and extract expense fields.

    Handles:
    - Real Pub/Sub push messages ({"message": {"data": "<base64>"}})
    - Direct payload with base64 data ({"data": "<base64>"})
    - Local testing plain JSON ({"data": {...}} or raw JSON string)
    """
    if hasattr(node_input, "parts"):
        # node_input is types.Content from START
        text = "".join(part.text for part in node_input.parts if part.text)
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return Event(output={"error": f"Invalid JSON string: {text[:200]}"})
    elif isinstance(node_input, str):
        try:
            event = json.loads(node_input)
        except json.JSONDecodeError:
            return Event(output={"error": f"Invalid JSON string: {node_input[:200]}"})
    elif isinstance(node_input, dict):
        event = node_input
    else:
        return Event(output={"error": f"Unsupported node input type: {type(node_input)}"})

    # If wrapped in Pub/Sub push format {"message": {"data": ...}}
    if "message" in event and isinstance(event["message"], dict):
        event = event["message"]

    data_field = event.get("data", event)

    if isinstance(data_field, str):
        # Could be base64-encoded (Pub/Sub) or a plain JSON string (local testing)
        decoded = None
        try:
            decoded_bytes = base64.b64decode(data_field, validate=True)
            decoded = json.loads(decoded_bytes.decode("utf-8"))
        except Exception:
            pass

        if decoded is None:
            try:
                decoded = json.loads(data_field)
            except Exception:
                return Event(output={"error": f"Failed to parse data field: {data_field[:200]}"})
        expense_dict = decoded
    elif isinstance(data_field, dict):
        expense_dict = data_field
    else:
        return Event(output={"error": f"Unexpected data field type: {type(data_field)}"})

    expense = ExpenseData(
        amount=float(expense_dict.get("amount", 0.0)),
        submitter=str(expense_dict.get("submitter", "unknown")),
        category=str(expense_dict.get("category", "other")),
        description=str(expense_dict.get("description", "")),
        date=str(expense_dict.get("date", "")),
    )

    return Event(output=expense.model_dump())


# ---------------------------------------------------------------------------
# Node 2: Routing based on dollar threshold (pure Python logic)
# ---------------------------------------------------------------------------


def route_by_amount(node_input: dict[str, Any], ctx: Context) -> Event:
    """Evaluate the dollar threshold and route the workflow in Python code.

    - Under $100 -> AUTO_APPROVE (no LLM involved)
    - $100 or more -> NEEDS_REVIEW (LLM review + HITL approval gate)
    """
    ctx.state["expense_data"] = node_input
    amount = float(node_input.get("amount", 0.0))

    if amount < config.review_threshold:
        return Event(route="AUTO_APPROVE", output=node_input)
    return Event(route="NEEDS_REVIEW", output=node_input)


# ---------------------------------------------------------------------------
# Branch A: Instant Auto-Approval (< $100)
# ---------------------------------------------------------------------------


def auto_approve(node_input: dict[str, Any]) -> Event:
    """Auto-approve low-value expenses immediately without LLM invocation."""
    amount = float(node_input.get("amount", 0.0))
    submitter = node_input.get("submitter", "unknown")
    description = node_input.get("description", "")

    log_entry = {
        "severity": "INFO",
        "decision": "approved",
        "method": "auto_approve",
        "message": f"Expense auto-approved: ${amount:.2f} from {submitter} ({description})",
        "amount": amount,
        "submitter": submitter,
        "category": node_input.get("category", ""),
        "date": node_input.get("date", ""),
    }
    print(json.dumps(log_entry), flush=True)

    outcome = {
        "status": "approved",
        "decision_method": "auto_approved",
        "expense": node_input,
    }
    return Event(output=outcome)


# ---------------------------------------------------------------------------
# Branch B: LLM Risk Assessment & Alert (>= $100)
# ---------------------------------------------------------------------------


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict[str, Any]:
    """Emit a structured warning log alerting finance/management of risk factors.

    Cloud Run captures this JSON log in Cloud Logging. Log-based alert policies
    notify managers via email or Pub/Sub.
    """
    log_entry = {
        "severity": "WARNING",
        "alert_type": "expense_review",
        "message": f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {
        "status": "alert_emitted",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }


review_agent = Agent(
    name="review_agent",
    model=config.model,
    mode="single_turn",
    instruction="""You are an expense compliance reviewer. You evaluate expense reports
of $100 or more that require risk evaluation before approval.

Analyze the submitted expense and:
1. Examine risk factors: excessive amount, unusual or mismatched category, vague or suspicious
   description, round figures, or policy violations.
2. Call the `emit_expense_alert` tool with the submitter, amount, category, and a concise risk
   summary explaining the key risks identified.
3. Return a structured assessment with:
   - Amount
   - Submitter
   - Category
   - Risk level: low, medium, or high
   - Risk factors: list of identified risk flags
   - Recommendation: approve or reject""",
    input_schema=ExpenseData,
    tools=[emit_expense_alert],
)


# ---------------------------------------------------------------------------
# Branch B (cont.): Human-in-the-Loop Pause & Decision Recording
# ---------------------------------------------------------------------------


def request_approval(node_input: Any, ctx: Context):
    """Pause the workflow using RequestInput until a human approves or rejects."""
    expense = ctx.state.get("expense_data", {})
    amount = float(expense.get("amount", 0.0))
    submitter = expense.get("submitter", "unknown")

    yield RequestInput(
        interrupt_id="expense_approval",
        message=(
            f"High-value expense (${amount:.2f} from {submitter}) flagged by risk review. "
            "Approve or reject."
        ),
        payload=expense,
    )


def process_decision(node_input: Any, ctx: Context) -> Event:
    """Process the human reviewer's decision and record the final outcome."""
    # node_input is the resume payload provided by the human reviewer:
    # e.g., {"decision": "approve"} / {"decision": "reject"} or string "approve" / "reject"
    decision_str = "unknown"
    if isinstance(node_input, dict):
        decision_str = str(node_input.get("decision", node_input.get("response", "unknown")))
    elif isinstance(node_input, str):
        decision_str = node_input

    approved = "approve" in decision_str.lower()
    status = "approved" if approved else "rejected"
    expense = ctx.state.get("expense_data", {})

    log_entry = {
        "severity": "INFO" if approved else "WARNING",
        "message": f"Expense {status} by human reviewer ({decision_str})",
        "decision": status,
        "decision_method": "human_review",
        "amount": expense.get("amount", 0.0),
        "submitter": expense.get("submitter", "unknown"),
    }
    print(json.dumps(log_entry), flush=True)

    outcome = {
        "status": status,
        "decision": decision_str,
        "decision_method": "human_review",
        "expense": expense,
    }
    return Event(output=outcome)


# ---------------------------------------------------------------------------
# Workflow Graph Topology
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="expense_approval_agent",
    edges=[
        ("START", parse_expense_event, route_by_amount),
        (
            route_by_amount,
            {
                "AUTO_APPROVE": auto_approve,
                "NEEDS_REVIEW": review_agent,
            },
        ),
        (review_agent, request_approval, process_decision),
    ],
)

app = App(
    name="expense_approval_agent",
    root_agent=root_agent,
)
