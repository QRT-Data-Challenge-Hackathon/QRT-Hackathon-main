# AGENTS.md — operating manual for every agent in this repo

Every AI agent (Claude Code, Codex, Cursor, Copilot, Gemini, …) and every human working here
follows this file. Claude-specific extras: [CLAUDE.md](CLAUDE.md). Event strategy and the
minute-by-minute plan: [docs/PLAYBOOK.md](docs/PLAYBOOK.md). Research behind it:
[reports/QRT Data Challenge hackathon format.md](reports/QRT%20Data%20Challenge%20hackathon%20format.md).

**Several agents work in this repo at the same time.** The rules below exist so that they never
overwrite each other, never break `main`, and never ship an unvalidated submission.

---

## 0. Hard rules (read these even if you read nothing else)

1. **Never work on `main`.** Never commit to it, never push to it, never merge into it locally.
   Every change reaches `main` only through a **squash-merged pull request**.
2. **One agent = one branch = one git worktree.** Start with `scripts/agent.sh start …` (§4).
   Never `git checkout`/`git switch` in a checkout another agent is using.
3. **Claim before you work:** check open PRs, then open a **draft PR** for your scope (§5).
   Do not edit files another open PR has claimed.
4. **Merge only through `scripts/safe_merge.sh <PR>`** (squash, CI green, no conflicts).
   You may merge your own PRs. You may merge another agent's PR only under §6.3.
5. **Never** force-push (except `--force-with-lease` on your own branch), never `--no-verify`,
   never `--admin`, never edit `.githooks/`, `.claude/hooks/` or `.claude/settings.json` to
   weaken them, never change the default branch, never delete branches you do not own.
6. **The contract files are frozen during the event** (§7.1): `competition.toml`,
   `src/qrtc/{cv,metrics,submit,io,config}.py`. Only the Validator's PRs change them.
7. **Only the Captain uploads submissions.** Agents write predictions to disk; they never
   submit, and never write "the final CSV".
8. **Never commit data, secrets, or notebook outputs.** `data/`, `runs/`, `submissions/*.csv`
   are gitignored; hooks reject notebooks with outputs and files > 2 MB.

GitHub branch protection is **not available** on this private free-plan repo. The guards are
local: `.githooks/` (pre-commit, pre-push; enabled by `make setup`) and the Claude Code
PreToolUse hook `.claude/hooks/guard_main.py`. They are defence in depth; these rules are the
real gate. If a guard blocks you, **stop and report it** — never rephrase a command to slip past.

---

## 1. Mission

**Event:** QRT Data Challenge Hackathon (QRT × Analytics Club ETH), ETH Zürich,
**Tue 24 Nov 2026, 18:00–22:00**, in person. Problem, metric, team size, platform and AI rules
are **unknown until the briefing**.

**Most likely (inference from QRT's 7 ENS challenges and past campus events):** one anonymised
table (`X_train`, `y_train`, `X_test`), upload `ID,prediction` CSV, hidden test set with a
public/private split, metric most likely **accuracy on a return sign** (else Spearman,
multiclass accuracy, IPCW C-index). Alternative: QRT's **"C3"** format — HackerRank
optimisation problems with partial credit (fallback kit in `src/qrtc/optim.py`,
`templates/optim/`).

**What wins (documented):** validation that groups by date, per-date cross-sectional features,
entity id as a categorical, target shaped to the metric, a small equal-weight blend, an early
valid submission, and **suspicion of every big gain**. In finance tasks the honest edge is
1–2 accuracy points — less than public-leaderboard noise — so **select on CV, never on the
public LB**.

---

## 2. Setup and commands

```bash
make setup                 # uv sync, git hooks on, Jupyter kernel "Python (qrtc)"
make check                 # ruff + pytest + notebook-output check (run before every PR)
make rehearse              # generate all 5 synthetic rehearsal datasets into data/synthetic/
make synth KIND=allocation # one rehearsal dataset (stock_sign allocation electricity football survival)

uv run qrtc eda                         # structure, split, target, per-date IC, leak flags
uv run qrtc baseline --model lgbm       # CV + save run + write validated submission
uv run qrtc lb                          # team leaderboard (CV, public LB, agent, branch, local)
uv run qrtc blend RUN1 RUN2 --method hill
uv run qrtc check-sub submissions/<file>.csv
uv run qrtc record-lb <run_id> <public_score>
QRTC_CONFIG=configs/rehearsal/football.toml uv run qrtc baseline   # rehearsal config
make lab                                # Jupyter Lab
```

Models (`--model`): `lgbm xgb catboost logreg ridge extratrees rf hgb`, survival:
`coxph coxnet rsf gbsa xgb`, presets: `lgbm_shallow lgbm_deep xgb_shallow catboost_shallow`.

---

## 3. Repository map

```
competition.toml          THE problem spec (files, id/target/date/group, task, metric, CV)
configs/rehearsal/        specs for the 5 synthetic shapes of past QRT challenges
src/qrtc/                 shared library (tested): config io cv metrics features models train
                          tracking ensemble postprocess submit eda synthetic optim cli
notebooks/00-03_*.ipynb   quickstart, EDA, experiment template, blend & submit (no outputs!)
experiments/<agent>/      YOUR notebooks/scripts; only you edit your folder
templates/optim/          C3 fallback: stdin/stdout Python template, C++ annealing template
tests/                    pytest (unit + integration + hook tests)
results/registry/, lb/    run metadata / public scores, committed via PRs (tiny JSON)
runs/, submissions/, data/  local only, gitignored, live in the MAIN checkout (shared by worktrees)
docs/PLAYBOOK.md          event plan, roles, briefing questions, recipes, leak table
reports/, research_notes/ the research on QRT's format (read-only)
scripts/                  agent.sh (worktrees), safe_merge.sh, setup.sh, check_notebooks.py
.githooks/, .claude/      guards (do not weaken)
```

Data paths in `competition.toml` resolve against the **main checkout**, so every worktree reads
the same `data/` and every agent's runs land in the same `runs/` (blendable by anyone).

---

## 4. Agent lifecycle (worktrees)

```bash
export AGENT_NAME=<you>                      # e.g. dmr-claude-1, cg-codex-2 (unique per agent)
scripts/agent.sh start feat rank-features    # worktree .worktrees/<agent>-rank-features on
                                             # branch feat/<agent>/rank-features from origin/main
cd .worktrees/<agent>-rank-features
# ... work, commit early and often (conventional commits) ...
scripts/agent.sh sync                        # fetch + merge origin/main into your branch
make check
git push -u origin HEAD && gh pr ready <n>   # or: gh pr create --fill (if not claimed yet)
scripts/safe_merge.sh <n>                    # squash-merge once CI is green
scripts/agent.sh finish                      # remove your worktree after the merge
```

- Branch names: `<type>/<agent>/<topic>`, type ∈ `feat fix exp docs chore refactor test`.
- Commit format: `<type>: <description>` (feat, fix, refactor, docs, test, chore, perf, ci).
- Keep PRs **small and short-lived** (≤ ~200 lines, ≤ ~30 min at the event). Merge, then
  start the next branch from fresh `origin/main`.
- Sync with `main` by **merging** `origin/main` into your branch (`scripts/agent.sh sync`), not
  by rebasing, so nobody needs to force-push.

---

## 5. Coordination protocol (multiple agents at once)

1. **Look before you start:** `gh pr list --state open` and `uv run qrtc lb`. If someone already
   claims your idea or files, pick something else or comment on their PR.
2. **Claim:** `scripts/agent.sh start … --claim "<scope: files/idea>"` pushes an empty commit and
   opens a **draft PR** titled `[<agent>] <type>: <topic>` whose body lists the files you will
   touch. The draft PR is the lock. Release it by merging, or by closing it with a comment.
3. **Ownership zones:**
   | Zone | Who may edit |
   |---|---|
   | `experiments/<agent>/`, `notebooks/scratch/<agent>/` | only that agent |
   | `src/qrtc/features.py`, `models.py`, `ensemble.py`, `postprocess.py`, `eda.py` | anyone, via a claimed small PR with tests |
   | contract files (§7.1), `notebooks/0*.ipynb`, `configs/` | Validator (event) / anyone with a claimed PR (before the event) |
   | `AGENTS.md`, `CLAUDE.md`, guards, CI, `scripts/` | only when the user asks |
4. **Add, don't rewrite:** new features/models go in new functions; do not change the signature
   or behaviour of a function others call without updating every caller and its tests in the
   same PR.
5. **Results are append-only:** every experiment is saved with `tracking.save_run` (unique
   `run_id` = time + agent + slug). Never delete or overwrite another agent's run, registry entry
   or submission file. Commit your `results/registry/*.json` in your PR.
6. **Talk through PRs:** progress, CV numbers and hand-offs go in PR descriptions/comments
   (`gh pr comment <n> -b "…"`). A PR description must contain: what changed, CV mean ± std vs
   the baseline run id, and the files touched.
7. **Conflicts:** first merged wins. The second agent runs `scripts/agent.sh sync`, resolves by
   keeping **both** sides' intent, re-runs `make check`, and pushes. Never resolve a conflict by
   deleting someone else's work. If you cannot resolve it, comment on both PRs and stop.
8. **After any merge to `main`:** every agent runs `scripts/agent.sh sync` before its next commit.

---

## 6. Pull requests and merging

### 6.1 Ready to merge (own PR)
- `make check` green locally; CI green on the PR; no merge conflicts with `main`.
- PR is not a draft; description complete (§5.6); no unresolved review comments.
- For model/feature PRs: the claimed CV gain is reproducible from the saved run and is larger
  than ~1 fold-std (otherwise it is noise — merge only the code, not the claim).

### 6.2 How
`scripts/safe_merge.sh <n>` — refuses drafts, conflicts, failing/pending CI, PRs not targeting
`main`, and PRs labelled `wip` / `do-not-merge`; then `gh pr merge <n> --squash --delete-branch`.
The raw command is allowed only in exactly that form (`--squash`, never `--admin`/`--merge`/
`--rebase`/`--auto`). If the repo has no CI workflow yet or GitHub Actions is down,
`--allow-no-ci` replaces CI with `make check` on a local checkout of the PR head (must equal
origin) — never use it to skip a red CI.

### 6.3 Merging another agent's PR (cautiously)
Allowed only when **all** hold:
- it is not a draft and not labelled `wip`/`do-not-merge`, and CI is green;
- the author marked it ready **and** has been inactive on it for ≥ 10 min (≥ 3 min at the
  event), **or** the author/user asked for it to be merged;
- you read the full diff, it does what the description says, it stays in its claimed scope,
  and it does not touch contract files (§7.1) unless the author is the Validator;
- merging it cannot break your own or others' open work (shared function signatures intact).
Then run `scripts/safe_merge.sh <n> --others "<one-line reason>"` (it comments the reason on the
PR). **When in doubt, comment instead of merging.**

---

## 7. Modelling rules (the event)

### 7.1 The contract (frozen at the event; Validator owns it)
`competition.toml` (spec, CV strategy, metric), `src/qrtc/cv.py`, `metrics.py`, `submit.py`,
`io.py`, `config.py`. Everyone uses **the same splits, seed and metric** — that is what makes
CV numbers from different agents comparable and blendable.

### 7.2 Validation discipline
- Use `qrtc.cv.make_splits` (date-grouped / time / group). **Never plain KFold on panel data.**
- Fit every scaler, encoder, imputer and target encoding **inside the folds**
  (`features.oof_target_encode`, sklearn pipelines in `models.py`).
- Report **CV mean ± std over folds** and the baseline run id. A gain smaller than ~1 std is noise.
- A single feature that adds > ~1 accuracy point on a sign task, or a CV far above the public
  LB, is **a leak until the Validator clears it**. Run `qrtc eda` leak flags,
  `eda.adversarial_validation`, `eda.lag_chain_links`.
- **Hidden structure** (recoverable date order, overlapping lags, ID order): detect it, report it,
  and **do not exploit it unless the organisers explicitly allowed it** at the briefing.
- External data: assume **forbidden** unless the organisers say otherwise.
- Select finals on CV, not on the public LB (shake-ups of 1st → 81st happened in QRT contests).

### 7.3 Experiment hygiene
- One idea per run, one change vs. the baseline. Use `notebooks/02_experiment_template.ipynb`
  copied into `experiments/<agent>/`.
- Every run through `train.run_cv` + `tracking.save_run` (OOF + test preds on disk, metadata in
  the registry). No untracked numbers in chat.
- Seed everything (`spec.seed`); never use wall-clock or row order as a feature.

---

## 8. Code rules

- Python 3.12 (`uv`), type hints, functions < 50 lines, files < 800 lines, no deep nesting.
- **Pure functions** for features: return new DataFrames, never mutate inputs.
- New shared code in `src/qrtc/` needs tests in `tests/`; keep `make check` green and coverage
  ≥ 80%. Notebooks/experiments are exempt from tests, not from the contract.
- Notebooks are committed **without outputs** (`uv run python scripts/check_notebooks.py --fix`).
- No `print` debugging left in `src/` except CLI output; errors are raised with clear messages.
- Don't `import torch` in the same process before LightGBM (known crash); on macOS LightGBM may
  need `brew install libomp`.

---

## 9. Event roles (details and clock in docs/PLAYBOOK.md)

| Role | Owns | Never |
|---|---|---|
| **Captain** (human) | data drop, uploads, `record-lb`, final selection, the clock | lets an agent upload |
| **Validator** (human + 1 agent) | contract files, leak/shift audits, approving claimed gains | approves own gains |
| **Lane owners** (humans, 2–3 agents each) | one written task per agent, e.g. "per-date ranks of RET_1..5, report CV vs run X" | "improve the score" prompts |

Milestones: trivial valid entry by T+30 min · date-grouped baseline by T+55 · lanes until
T+135 · blend + two finals by T+155 · **code freeze at T−20**.

---

## 10. Never

- Commit/push/merge on `main`, force-push, `--no-verify`, `--admin`, or weaken a guard.
- Merge a draft, a `wip`/`do-not-merge` PR, a red-CI PR, or a PR you have not read.
- Edit another agent's claimed files, experiments folder, runs or submissions.
- Upload a submission (Captain only) or one that failed `qrtc check-sub`.
- Exploit leaks or use external data without explicit organiser permission.
- Commit organiser data, secrets, `.env`, or notebooks with outputs.
