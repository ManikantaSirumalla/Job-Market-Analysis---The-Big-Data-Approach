
.PHONY: api test format setup

# Set PYTHONPATH for all commands
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

format:
	black src

setup:
	@echo "Setting up environment..."
	@python3.11 -m venv .venv
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p data/raw data/bronze data/silver data/gold configs tests
	@echo "Environment setup complete!"
	@echo "Run 'source activate_env.sh' to activate the environment"
