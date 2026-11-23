# CLAUDE.md

Instructions for Claude Code in this repository. The shared rules for all agents (git, multi-agent
coordination, merging, modelling discipline, event roles) live in AGENTS.md and are imported here:

@AGENTS.md

## Claude-specific

### Guard hook

`.claude/hooks/guard_main.py` runs before every Bash call. It blocks: commits/merges/rebases on
`main`; pushes to `main` (explicit, implicit, `HEAD`, `--all`, `--mirror`, deletes); force pushes
(`-f`, `--force`, `+ref`); `--no-verify`; `gh pr merge` unless it is a plain `--squash` merge
(no `--admin`, `--auto`, `--merge`, `--rebase`); branch-protection / ruleset / merge mutations via
`gh api`; `gh repo delete`; default-branch changes. If it blocks you, **stop and tell the user**.
Never rephrase a command to slip past it, and never edit the hook, `.githooks/` or
`.claude/settings.json` to weaken them.

### Starting work

1. `echo $AGENT_NAME` — if empty, pick a unique name (`<user>-claude-<n>`) and export it.
2. `git -C <repo> rev-parse --abbrev-ref HEAD` — if you are on `main` in the shared checkout,
   do **not** switch branches there: `scripts/agent.sh start <type> <topic> --claim "<scope>"`
   and work inside the printed `.worktrees/...` directory (or use the `EnterWorktree` tool with
   the same branch naming).
3. `gh pr list --state open` — respect other agents' claims (AGENTS.md §5).

### Skills — when to use what

| Situation | Skill |
|---|---|
| New feature/idea before coding (pre-event) | `superpowers:brainstorming` (at the event: 2-minute version) |
| Multi-step build (pre-event) | `superpowers:writing-plans` → `superpowers:executing-plans` |
| Shared code in `src/qrtc/` | `superpowers:test-driven-development`, `python-testing`, `python-patterns` |
| Strange score / bug | `superpowers:systematic-debugging` before changing anything |
| Several independent ideas | `superpowers:dispatching-parallel-agents` (one idea per agent, one worktree each) |
| Isolation | `superpowers:using-git-worktrees` (branch names per AGENTS.md §4) |
| Before claiming done / opening a PR | `superpowers:verification-before-completion`, `verification-loop` |
| Review | `superpowers:requesting-code-review`, `code-review`; reviewer ≠ author |
| Wrapping up a branch | `superpowers:finishing-a-development-branch` → choose "open PR", then `scripts/safe_merge.sh` |
| Backtest/strategy sanity (if the task is a trading/Sharpe one) | `antigravity-quant-analyst` |
| Charts in notebooks | `dataviz` |
| Library/API questions | context7 / vendor docs |

**Do not use for this project:** `academic-research-skills:*`, `finance:*` accounting skills,
brokerage/order tools of any MCP.

### Working style at the event

- Speed over ceremony, never over the contract: no plan documents during the contest, but every
  experiment still goes through `run_cv` + `save_run`, and every PR through `make check`.
- Report results as `run_id: metric mean ± std (baseline run_id: …)` in the PR, not only in chat.
- Write findings about the competition (rules from the briefing, data quirks, leaks) into
  `docs/PLAYBOOK.md` §Event log via a small PR.
- Never upload to the competition platform; hand the Captain the `submissions/...csv` path.
