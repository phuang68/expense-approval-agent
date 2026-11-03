.PHONY: install serve playground test lint

install:
	uv sync --all-extras

serve:
	uv run python -m app.fast_api_app

playground:
	uv run adk web . --host 127.0.0.1 --port 8080 --allow_origins "*" --reload_agents --trigger_sources pubsub

test:
	uv run pytest tests/unit

lint:
	uv run --extra lint ruff check app expense_approval_agent tests
