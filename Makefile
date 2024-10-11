.PHONY: install check predict clean runner
.DEFAULT_GOAL:=runner

install: pyproject.toml
	poetry install

check: install
	ruff check src

predict:
	poetry run python src/inference.py

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

runner: check predict clean
