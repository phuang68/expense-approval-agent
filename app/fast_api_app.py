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

"""Ambient FastAPI web service for the expense approval agent.

Accepts Pub/Sub push messages on /apps/{app_name}/trigger/pubsub and feeds
each message into the ADK 2.0 graph workflow.
"""

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from starlette.middleware.base import BaseHTTPMiddleware

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes

# Standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("expense_approval_service")

load_dotenv()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Telemetry: Explicitly set otel_to_cloud=False
otel_to_cloud = False

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PubSubSubscriptionNormalizerMiddleware(BaseHTTPMiddleware):
    """Normalize fully-qualified Pub/Sub subscription paths down to short names.

    Pub/Sub push delivers subscription names formatted as:
        projects/{project-id}/subscriptions/{subscription-name}
    ADK uses this subscription as the session user_id. Normalizing to the
    short subscription-name keeps session records concise and human-readable.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and "/trigger/pubsub" in request.url.path:
            body = await request.body()
            try:
                data = json.loads(body)
                if (
                    isinstance(data, dict)
                    and "subscription" in data
                    and isinstance(data["subscription"], str)
                ):
                    raw_sub = data["subscription"]
                    normalized_sub = raw_sub.split("/")[-1]
                    if raw_sub != normalized_sub:
                        logger.info(
                            "Normalizing subscription path: '%s' -> '%s'",
                            raw_sub,
                            normalized_sub,
                        )
                        data["subscription"] = normalized_sub
                        new_body = json.dumps(data).encode("utf-8")

                        async def receive():
                            return {"type": "http.request", "body": new_body}

                        request = Request(request.scope, receive=receive)
            except Exception as e:
                logger.warning("Failed to normalize Pub/Sub body: %s", e)

        return await call_next(request)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    trigger_sources=["pubsub"],
    lifespan=lifespan,
)
app.add_middleware(PubSubSubscriptionNormalizerMiddleware)

app.title = "expense-approval-agent"
app.description = "Ambient API for interacting with the Agent expense-approval-agent"


# Main execution
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ambient expense approval agent on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
