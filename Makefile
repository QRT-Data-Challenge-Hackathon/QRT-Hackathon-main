# Run `make help` for the targets.
MODEL ?= lgbm

.PHONY: help setup models check lint test nbcheck nbstrip prep features eda baseline ridge lb lab clean-runs

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

setup:           ## dependencies, git hooks, Jupyter kernel (once per clone)
	@scripts/setup.sh

models:          ## optional pretrained TabICL/TabPFN/TabDPT: install, download weights, offline check
	uv sync --all-groups --extra fm
	uv run python scripts/prefetch_models.py

check: lint test nbcheck  ## what CI runs

lint:            ## ruff
	uv run ruff check .

test:            ## pytest with coverage
	uv run pytest -q --cov=src/qrtc --cov-report=term-missing:skip-covered --cov-fail-under=80

nbcheck:         ## fail if any notebook has outputs
	uv run python scripts/check_notebooks.py

nbstrip:         ## strip outputs from all notebooks
	uv run python scripts/check_notebooks.py --fix

prep:            ## flat baseline features (data/raw/.../prep/)
	uv run python scripts/prep_last30.py

features:        ## intraday feature tables (data/derived/)
	uv run python experiments/intraday_ridge/build.py

eda:             ## EDA on competition.toml (or QRTC_CONFIG)
	uv run qrtc eda

baseline:        ## CV baseline + saved run + submission: make baseline MODEL=lgbm_shallow
	uv run qrtc baseline --model $(MODEL)

ridge:           ## the ridge model on the intraday features (needs `make features`)
	QRTC_CONFIG=experiments/intraday_ridge/spec.toml uv run python experiments/intraday_ridge/run.py --model ridge --regress --no-fit-weight --slug ridge-v1

lb:              ## every saved run, best CV first
	uv run qrtc lb

lab:             ## Jupyter Lab
	uv run jupyter lab

clean-runs:      ## delete local runs/ and submissions (asks first)
	@read -p "delete runs/ and submissions/*.csv? [y/N] " a && [ "$$a" = y ] && rm -rf runs && find submissions -name '*.csv' -delete || true
