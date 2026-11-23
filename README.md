# QRT-Hackathon-main

Toolkit and operating rules for the **QRT Data Challenge Hackathon** (QRT × Analytics Club ETH),
ETH Zürich, **Tue 24 Nov 2026, 18:00–22:00**.

| Read | For |
|---|---|
| [docs/PLAYBOOK.md](docs/PLAYBOOK.md) | what to expect, the clock, briefing questions, recipes, leak reflexes |
| [AGENTS.md](AGENTS.md) | rules for every agent/human: protect `main`, worktrees, claims, PRs, merging |
| [reports/QRT Data Challenge hackathon format.md](reports/QRT%20Data%20Challenge%20hackathon%20format.md) | the research (QRT's 7 ENS challenges, past campus events, winning approaches) |

## Quick start

```bash
make setup                       # uv deps (Python 3.12), git guard hooks, Jupyter kernel
make models                      # pretrained TabICL/TabPFN/TabDPT weights + offline check
export AGENT_NAME=<unique-name>  # every agent/human gets one
make rehearse                    # synthetic datasets shaped like past QRT challenges
make check                       # lint + tests + notebook check
uv run qrtc eda                  # on competition.toml (rehearsal stock_sign by default)
uv run qrtc baseline --model lgbm_shallow
uv run qrtc lb
uv run qrtc upload submissions/<file>.csv -m "why"   # designated uploader only (AGENTS.md §9.1)
```

At the event: put organiser files in `data/raw/`, edit `competition.toml` (start from the closest
`configs/rehearsal/*.toml`), merge it in one PR, then work in lanes
(`scripts/agent.sh start feat <topic> --claim "<scope>"`).

## What is in here

- `src/qrtc/` — tested library: spec loading, data loading, leak-safe CV (date blocks, time,
  group), 16 metrics (accuracy … IPCW C-index), feature helpers (per-date ranks/z-scores, group
  de-meaning, OOF target encoding, rank target), model zoo (LightGBM, XGBoost, CatBoost, linear,
  trees, survival), CV loop with OOF/test predictions, shared run registry, hill-climb/rank
  blending, threshold/median-split post-processing, validated submission writer, EDA (per-date
  IC, adversarial validation, leak checks), rehearsal data generators, optimisation engines.
- `notebooks/` — 00 quickstart, 01 EDA, 02 experiment template, 03 blend & submit.
- `templates/optim/` — fallback for QRT's "C3" optimisation format (Python + C++).
- Guards: `.githooks/` (no commits/pushes on main, no data/outputs) and
  `.claude/hooks/guard_main.py` (Claude Code). Merges: `scripts/safe_merge.sh <PR>` only.
