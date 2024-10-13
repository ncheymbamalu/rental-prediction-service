.PHONY: install check train predict clean runner_train runner_predict
.DEFAULT_GOAL:=runner_predict

install: pyproject.toml
	poetry install

check: install
	poetry run ruff check src

train:
	poetry run python src/run_model_builder.py

predict:
	poetry run python src/run_model_inference.py

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

runner_train: check train clean

runner_predict: check predict clean
