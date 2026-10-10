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

"""Unit tests for the ambient expense approval agent workflow."""

import base64
import json
from unittest.mock import MagicMock

import pytest
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events import Event, RequestInput
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow
from google.genai import types

from expense_approval_agent.agent import (
    auto_approve,
    emit_expense_alert,
    parse_expense_event,
    process_decision,
    request_approval,
    route_by_amount,
)
from expense_approval_agent.config import config


def test_config_values():
    """Verify that model and threshold are loaded properly in config."""
    assert config.model == "gemini-3.1-flash-lite"
    assert config.review_threshold == 100.0


def test_parse_expense_event_plain_dict():
    """Test parsing a plain JSON dictionary event."""
    event_dict = {
        "data": {
            "amount": 75.50,
            "submitter": "alice@example.com",
            "category": "Meals",
            "description": "Team lunch",
            "date": "2026-10-10",
        }
    }
    result = parse_expense_event(event_dict)
    assert isinstance(result, Event)
    output = result.output
    assert output["amount"] == 75.50
    assert output["submitter"] == "alice@example.com"
    assert output["category"] == "Meals"
    assert output["description"] == "Team lunch"
    assert output["date"] == "2026-10-10"


def test_parse_expense_event_base64_pubsub():
    """Test parsing a real Pub/Sub push notification with base64 encoded data."""
    raw_expense = {
        "amount": 250.00,
        "submitter": "bob@example.com",
        "category": "Travel",
        "description": "Flight to Chicago",
        "date": "2026-10-11",
    }
    b64_data = base64.b64encode(json.dumps(raw_expense).encode("utf-8")).decode("utf-8")
    pubsub_msg = {"message": {"data": b64_data, "messageId": "msg-999"}}

    result = parse_expense_event(pubsub_msg)
    assert isinstance(result, Event)
    output = result.output
    assert output["amount"] == 250.00
    assert output["submitter"] == "bob@example.com"
    assert output["description"] == "Flight to Chicago"


def test_parse_expense_event_types_content():
    """Test parsing when START node delivers types.Content."""
    raw_expense = {
        "data": {
            "amount": 42.00,
            "submitter": "carol@example.com",
            "category": "Supplies",
            "description": "Notebooks",
            "date": "2026-10-09",
        }
    }
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(raw_expense))],
    )
    result = parse_expense_event(content)
    assert isinstance(result, Event)
    assert result.output["amount"] == 42.00


def test_route_by_amount_under_threshold():
    """Test routing when expense is under $100 -> AUTO_APPROVE."""
    ctx = MagicMock(spec=Context)
    ctx.state = {}
    data = {"amount": 99.99, "submitter": "dan@example.com"}

    event = route_by_amount(data, ctx)
    assert event.actions.route == "AUTO_APPROVE"
    assert ctx.state["expense_data"] == data


def test_route_by_amount_at_or_above_threshold():
    """Test routing when expense is $100 or more -> NEEDS_REVIEW."""
    ctx = MagicMock(spec=Context)
    ctx.state = {}
    data = {"amount": 100.00, "submitter": "dan@example.com"}

    event = route_by_amount(data, ctx)
    assert event.actions.route == "NEEDS_REVIEW"
    assert ctx.state["expense_data"] == data

    data_high = {"amount": 500.00, "submitter": "dan@example.com"}
    event_high = route_by_amount(data_high, ctx)
    assert event_high.actions.route == "NEEDS_REVIEW"


def test_auto_approve():
    """Test auto-approval node output."""
    expense = {
        "amount": 50.00,
        "submitter": "eve@example.com",
        "category": "Meals",
        "description": "Coffee with client",
        "date": "2026-10-08",
    }
    event = auto_approve(expense)
    assert event.output["status"] == "approved"
    assert event.output["decision_method"] == "auto_approved"
    assert event.output["expense"]["amount"] == 50.00


def test_emit_expense_alert():
    """Test the alert tool."""
    res = emit_expense_alert(
        submitter="frank@example.com",
        amount=350.00,
        category="Travel",
        risk_summary="High accommodation expense",
    )
    assert res["status"] == "alert_emitted"
    assert res["amount"] == 350.00


def test_request_approval():
    """Test the HITL RequestInput generation."""
    ctx = MagicMock(spec=Context)
    ctx.state = {
        "expense_data": {
            "amount": 350.00,
            "submitter": "frank@example.com",
            "category": "Travel",
        }
    }
    generator = request_approval(None, ctx)
    req = next(generator)
    assert isinstance(req, RequestInput)
    assert req.interrupt_id == "expense_approval"
    assert req.payload["amount"] == 350.00


def test_process_decision_approved():
    """Test human approval decision processing."""
    ctx = MagicMock(spec=Context)
    ctx.state = {
        "expense_data": {
            "amount": 350.00,
            "submitter": "frank@example.com",
        }
    }
    res = process_decision({"decision": "approve"}, ctx)
    assert res.output["status"] == "approved"
    assert res.output["decision_method"] == "human_review"


def test_process_decision_rejected():
    """Test human rejection decision processing."""
    ctx = MagicMock(spec=Context)
    ctx.state = {
        "expense_data": {
            "amount": 350.00,
            "submitter": "frank@example.com",
        }
    }
    res = process_decision({"decision": "reject"}, ctx)
    assert res.output["status"] == "rejected"
    assert res.output["decision_method"] == "human_review"


@pytest.mark.asyncio
async def test_full_workflow_auto_approve_branch():
    """Test running the full workflow with an expense under $100.

    Ensures it completes in one turn without calling the LLM.
    """
    from expense_approval_agent.agent import root_agent

    app = App(name="test_app", root_agent=root_agent)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(app_name="test_app", user_id="user_1")

    payload = {
        "data": {
            "amount": 80.00,
            "submitter": "grace@example.com",
            "category": "Office Supplies",
            "description": "Printer ink",
            "date": "2026-10-10",
        }
    }
    msg = types.Content(role="user", parts=[types.Part.from_text(text=json.dumps(payload))])

    events = []
    async for event in runner.run_async(
        user_id="user_1", session_id=session.id, new_message=msg
    ):
        events.append(event)

    outputs = [e.output for e in events if getattr(e, "output", None)]
    assert len(outputs) >= 1
    final_output = outputs[-1]
    assert final_output["status"] == "approved"
    assert final_output["decision_method"] == "auto_approved"
    assert final_output["expense"]["amount"] == 80.00


@pytest.mark.asyncio
async def test_full_workflow_hitl_branch_with_mock_llm():
    """Test the >= $100 branch pausing at RequestInput and resuming upon human approval."""
    # Build a test workflow with a mock review agent node to run completely offline
    def mock_review_node(node_input: dict) -> Event:
        emit_expense_alert(
            submitter=node_input["submitter"],
            amount=node_input["amount"],
            category=node_input["category"],
            risk_summary="High expense flagged for review",
        )
        return Event(output=node_input)

    test_wf = Workflow(
        name="test_expense_approval",
        edges=[
            ("START", parse_expense_event, route_by_amount),
            (
                route_by_amount,
                {
                    "AUTO_APPROVE": auto_approve,
                    "NEEDS_REVIEW": mock_review_node,
                },
            ),
            (mock_review_node, request_approval, process_decision),
        ],
    )

    app = App(name="test_hitl_app", root_agent=test_wf)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(app_name="test_hitl_app", user_id="user_2")

    raw_expense = {
        "amount": 450.00,
        "submitter": "heidi@example.com",
        "category": "Travel",
        "description": "Hotel accommodation",
        "date": "2026-10-12",
    }
    b64_data = base64.b64encode(json.dumps(raw_expense).encode("utf-8")).decode("utf-8")
    payload = {"message": {"data": b64_data}}
    msg = types.Content(role="user", parts=[types.Part.from_text(text=json.dumps(payload))])

    # Turn 1: Process expense and pause on RequestInput
    turn1_events = []
    async for event in runner.run_async(
        user_id="user_2", session_id=session.id, new_message=msg
    ):
        turn1_events.append(event)

    # Verify that an interrupt event was generated
    has_interrupt = any(
        e.content
        and any(
            p.function_call and p.function_call.name == "adk_request_input"
            for p in e.content.parts
        )
        for e in turn1_events
        if getattr(e, "content", None)
    )
    assert has_interrupt, "Expected workflow to pause with adk_request_input"

    # Turn 2: Resume with human manager decision ("approve")
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_input",
                    id="expense_approval",
                    response={"decision": "approve"},
                )
            )
        ],
    )

    turn2_events = []
    async for event in runner.run_async(
        user_id="user_2", session_id=session.id, new_message=resume_msg
    ):
        turn2_events.append(event)

    turn2_outputs = [e.output for e in turn2_events if getattr(e, "output", None)]
    assert len(turn2_outputs) >= 1
    final_output = turn2_outputs[-1]
    assert final_output["status"] == "approved"
    assert final_output["decision_method"] == "human_review"
    assert final_output["expense"]["amount"] == 450.00
