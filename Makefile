.PHONY: install test

install:
	python -m pip install -e ".[dev]"

test:
	pytest
