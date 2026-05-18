# Convenience shortcuts — same as: python scripts/run.py <cmd>
# Usage: make predict TICKER=MSFT

PYTHON ?= .venv/bin/python
TICKER ?= MSFT
PERIOD ?= 5y

.PHONY: help setup check train predict export pipeline compare api ui

help:
	@echo "Targets: setup check train predict export pipeline api ui"
	@echo "First time: make setup   then edit .env with FINNHUB_API_KEY"
	@echo "Example: make predict TICKER=MSFT"

setup:
	$(PYTHON) scripts/run.py setup

check:
	$(PYTHON) scripts/run.py check $(TICKER)

train:
	$(PYTHON) scripts/run.py train $(TICKER) --period $(PERIOD)

predict:
	$(PYTHON) scripts/run.py predict $(TICKER) --compact

export:
	$(PYTHON) scripts/run.py export $(TICKER) -o features_$(TICKER).csv --period $(PERIOD)

pipeline:
	$(PYTHON) scripts/run.py pipeline $(TICKER) --period $(PERIOD) --compact

compare:
	$(PYTHON) scripts/run.py compare $(TICKER)

api:
	$(PYTHON) scripts/run.py api

ui:
	$(PYTHON) scripts/run.py ui
