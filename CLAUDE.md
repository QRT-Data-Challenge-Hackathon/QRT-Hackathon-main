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
`gh api`; `gh repo delete`; default-branch changes; `kaggle competitions submit`.
`.claude/hooks/guard_upload.py` runs before every browser `file_upload` and blocks files that
`uv run qrtc upload` has not reserved (sha256 must match a `pending` ledger entry). If a hook
blocks you, **stop and tell the user**.
Never rephrase a command to slip past it, and never edit the hook, `.githooks/` or
`.claude/settings.json` to weaken them.

### Starting work

1. `echo $AGENT_NAME` — if empty, pick a unique name (`<user>-claude-<n>`) and export it.
2. `git -C <repo> rev-parse --abbrev-ref HEAD` — if you are on `main` in the shared checkout,
   do **not** switch branches there: `scripts/agent.sh start <type> <topic> --claim "<scope>"`
   and work inside the printed `.worktrees/...` directory (or use the `EnterWorktree` tool with
   the same branch naming).
3. `gh pr list --state open` — respect other agents' claims (AGENTS.md §5).
4. `gh issue list --state all` — before starting any idea, check whether a coworker already has
   an issue for it. If so, pick something else or comment on their issue.
5. New idea → **open a GitHub issue first**, before any code:
   `gh issue create --title "[<AGENT_NAME>] exp: <topic>" --body "<idea, files, how measured>"`.
   Reference it in the branch's PR (`Refs #<n>` / `Closes #<n>`) and post results on it.

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
- Upload only if your `AGENT_NAME` is in `[upload].uploaders` (the Submitter, AGENTS.md §9.1);
  otherwise hand the Submitter the `submissions/...csv` path.
- Submitter with `backend = "browser"` (Claude in Chrome): a human logs in to the platform
  first — you never type credentials. Then `uv run qrtc upload <file> -m "…"` → read the page
  (`find` the file input) → `file_upload` with exactly the printed path → submit → read the
  public score → `uv run qrtc upload-confirm <id> --score <s>`. Claude asks the user in chat
  before clicking a platform's submit button (a standing browser-safety rule), so tell the
  Captain to watch the chat at upload milestones; `kaggle`/`http` backends need no click.
  If `file_upload` rejects the path as not shared with the session, `cp` the file into your
  worktree and upload the copy (the hook checks the sha256, not the path), or ask the user.
