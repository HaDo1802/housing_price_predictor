SHELL := /bin/bash

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SRC_PATHS := src app.py predict.py
TEST_PATHS ?= tests

.PHONY: help install lint format test clean all

help:
	@echo "Available commands:"
	@echo "  make install  - Install Python dependencies"
	@echo "  make test     - Run tests (template)"
	@echo "  make lint     - Run linting checks"
	@echo "  make format   - Format code with black and isort"
	@echo "  make clean    - Remove generated files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(SRC_PATHS)

format:
	$(PYTHON) -m black $(SRC_PATHS)
	$(PYTHON) -m isort $(SRC_PATHS)

test:
	@if [ -d "$(TEST_PATHS)" ]; then $(PYTHON) -m pytest $(TEST_PATHS) -v; else echo "No $(TEST_PATHS) directory yet."; fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

all: install format lint test
