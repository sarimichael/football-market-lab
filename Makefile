.PHONY: install demo data baseline test lint all

install:
	pip install -e ".[dev]"

demo:
	python -m fml demo

data:
	python -m fml download
	python -m fml build

baseline:
	python -m fml baseline

test:
	pytest

lint:
	ruff check .
	ruff format --check .

all: lint test data baseline
