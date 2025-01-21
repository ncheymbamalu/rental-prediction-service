.PHONY: install check data train predict backend clean runner_train runner_predict runner_backend
.DEFAULT_GOAL:=runner_backend

install: pyproject.toml
	poetry install

check: install
	poetry run ruff check src

data: check
	poetry run python src/database.py

train:
	poetry run python src/run_model_builder.py

predict:
	poetry run python src/run_model_inference.py

backend:
	uvicorn src.app:app --reload

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

runner_train: check train clean

runner_predict: check predict clean

runner_backend: check backend clean
