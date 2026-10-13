.PHONY: install playground test lint

install:
	uv sync --all-extras

playground:
	@if command -v agents-cli >/dev/null 2>&1; then \
		agents-cli playground --port 8080; \
	else \
		uv run adk web . --host 127.0.0.1 --port 8080 --allow_origins "*" --reload_agents; \
	fi

test:
	uv run pytest tests/unit

lint:
	uv run --extra lint ruff check .
