.PHONY: install test mock demo bench report

install:
	uv venv && uv pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

mock:
	.venv/bin/tokenbench mock

# Full pipeline against the mock server (start `make mock` in another terminal first)
demo:
	.venv/bin/tokenbench all -c configs/mock.yaml -o runs/mock-demo --levels 1,2,4,8,16 --max-tokens 64

# Full pipeline against Nebius Token Factory (needs NEBIUS_API_KEY)
bench:
	.venv/bin/tokenbench all -c configs/nebius.yaml -o runs/nebius
