SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SRC := src app.py predict.py
TEST_DIR := tests

.PHONY: help install lint format test clean all

help:
	@printf "Targets:\\n"
	@printf "  install  Install Python dependencies\\n"
	@printf "  lint     Run flake8\\n"
	@printf "  format   Run black + isort\\n"
	@printf "  test     Run pytest if a test dir exists\\n"
	
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(SRC) --max-line-length 100 --extend-ignore=E501,E402,F401,F541,W291,W293
	

format:
	$(PYTHON) -m black $(SRC)
	isort $(SRC)

test:
	@if [ -n "$(TEST_DIR)" ]; then $(PYTHON) -m pytest $(TEST_DIR) -v; else echo "No test/ or tests/ directory found."; fi


all: install format lint test
