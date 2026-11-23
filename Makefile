# QRT Data Challenge toolkit. Run `make help`.
KIND ?= stock_sign
MODEL ?= lgbm
KINDS := stock_sign allocation electricity football survival

.PHONY: help setup check lint test nbcheck nbstrip synth rehearse eda baseline lb lab status clean-runs

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

setup:           ## deps, git guard hooks, Jupyter kernel (once per clone)
	@scripts/setup.sh

check: lint test nbcheck  ## everything CI runs; must be green before a PR

lint:            ## ruff
	uv run ruff check .

test:            ## pytest with coverage
	uv run pytest -q --cov=src/qrtc --cov-report=term-missing:skip-covered --cov-fail-under=80

nbcheck:         ## fail if any notebook has outputs
	uv run python scripts/check_notebooks.py

nbstrip:         ## strip outputs from all notebooks
	uv run python scripts/check_notebooks.py --fix

synth:           ## one rehearsal dataset: make synth KIND=allocation
	uv run qrtc synth $(KIND)

rehearse:        ## all rehearsal datasets
	@for k in $(KINDS); do uv run qrtc synth $$k; done

eda:             ## EDA on competition.toml (or QRTC_CONFIG)
	uv run qrtc eda

baseline:        ## CV baseline + run + submission: make baseline MODEL=lgbm_shallow
	uv run qrtc baseline --model $(MODEL)

lb:              ## team leaderboard
	uv run qrtc lb

lab:             ## Jupyter Lab
	uv run jupyter lab

status:          ## worktrees and open PRs (who is doing what)
	@scripts/agent.sh status

clean-runs:      ## delete local runs/ and submissions (asks first)
	@read -p "delete runs/ and submissions/*.csv? [y/N] " a && [ "$$a" = y ] && rm -rf runs && find submissions -name '*.csv' -delete || true
