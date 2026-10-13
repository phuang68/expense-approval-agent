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

"""Configuration settings for the ambient expense approval agent."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# --- Authentication Configuration ---
# If GOOGLE_API_KEY is provided, use Google AI Studio.
# Otherwise, default to Vertex AI with application default credentials.
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    try:
        import google.auth

        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id or "")
    except Exception:
        pass
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


@dataclass
class ExpenseApprovalConfig:
    """Agent configuration for model selection and approval threshold."""

    # gemini-3.1-flash-lite as requested by user
    model: str = os.getenv("EXPENSE_MODEL", "gemini-3.1-flash-lite")
    # $100 threshold for automatic approval vs manual risk review
    review_threshold: float = float(os.getenv("EXPENSE_THRESHOLD", "100.0"))


config = ExpenseApprovalConfig()
