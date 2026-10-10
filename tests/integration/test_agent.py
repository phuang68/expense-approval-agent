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

"""Integration test for the expense approval agent."""

import json

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_agent_stream() -> None:
    """Integration test for the expense agent streaming workflow.

    Verifies that an incoming expense event runs through the graph workflow.
    """
    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    payload = {
        "data": {
            "amount": 49.99,
            "submitter": "tester@example.com",
            "category": "Office Supplies",
            "description": "Sticky notes",
            "date": "2026-10-10",
        }
    }
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(payload))]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one event from workflow"

    outputs = [e.output for e in events if getattr(e, "output", None)]
    assert len(outputs) > 0, "Expected workflow output event"
    assert outputs[-1]["status"] == "approved"
    assert outputs[-1]["decision_method"] == "auto_approved"
