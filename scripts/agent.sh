#!/usr/bin/env bash
# Agent worktree lifecycle (AGENTS.md §4-5). One agent = one branch = one worktree.
#
#   scripts/agent.sh start <type> <topic> [--claim "<scope: files / idea>"]
#   scripts/agent.sh sync       # merge fresh origin/main into the current branch
#   scripts/agent.sh finish     # remove this worktree once its PR is merged/closed
#   scripts/agent.sh status     # worktrees + open PRs (who is doing what)
#
# Requires AGENT_NAME (unique per agent, e.g. dmr-claude-1). Branch: <type>/<agent>/<topic>.
set -euo pipefail

TYPES="feat fix exp docs chore refactor test"
ROOT="$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"

die() { echo "agent.sh: $*" >&2; exit 1; }
slug() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'; }
agent() { slug "${AGENT_NAME:?set AGENT_NAME first, e.g. export AGENT_NAME=dmr-claude-1}"; }

cmd_start() {
  [ $# -ge 2 ] || die "usage: start <type> <topic> [--claim \"scope\"]"
  local type=$1 topic; topic=$(slug "$2"); shift 2
  local claim=""
  while [ $# -gt 0 ]; do
    case $1 in --claim) claim=${2:?--claim needs a description}; shift 2 ;; *) die "unknown option $1" ;; esac
  done
  case " $TYPES " in *" $type "*) ;; *) die "type must be one of: $TYPES" ;; esac
  local who branch dir
  who=$(agent); branch="$type/$who/$topic"; dir="$ROOT/.worktrees/$who-$topic"
  [ -e "$dir" ] && die "$dir already exists"
  git -C "$ROOT" fetch -q origin main
  git -C "$ROOT" worktree add -q --no-track -b "$branch" "$dir" origin/main
  local fm="--extra fm"; [ -z "${QRTC_NO_FM:-}" ] || fm=""   # QRTC_NO_FM=1: skip torch models
  (cd "$dir" && uv sync -q --all-groups $fm)
  if [ -n "$claim" ]; then
    git -C "$dir" commit -q --allow-empty -m "chore: claim $topic" -m "$claim"
    git -C "$dir" push -q -u origin "$branch"
    (cd "$dir" && gh pr create --draft --base main --head "$branch" \
      --title "[$who] $type: $topic" \
      --body "$(printf '**Claim (draft = lock).** Agent: `%s`\n\nScope / files:\n%s\n\n_Mark ready when done; include CV mean ± std vs baseline run id._' "$who" "$claim")")
  fi
  echo "worktree ready on $branch"
  echo "cd $dir"
}

cmd_sync() {
  local branch; branch=$(git rev-parse --abbrev-ref HEAD)
  git fetch -q origin main
  if [ "$branch" = "main" ]; then
    git merge --ff-only origin/main
  else
    git merge --no-edit origin/main || die "merge conflict: resolve keeping BOTH sides' intent (AGENTS.md §5.7), then make check"
  fi
  echo "synced $branch with origin/main"
}

cmd_finish() {
  local dir branch state
  dir=$(git rev-parse --show-toplevel); branch=$(git rev-parse --abbrev-ref HEAD)
  [ "$dir" != "$ROOT" ] || die "run finish inside an agent worktree, not the main checkout"
  state=$(gh pr view "$branch" --json state -q .state 2>/dev/null || echo NONE)
  case $state in
    MERGED|CLOSED) ;;
    *) [ "${1:-}" = "--force" ] || die "PR for $branch is $state; merge/close it first (or --force)";;
  esac
  [ -z "$(git status --porcelain)" ] || [ "${1:-}" = "--force" ] || die "uncommitted changes in $dir"
  cd "$ROOT"
  git worktree remove ${1:+--force} "$dir"
  git branch -D "$branch" >/dev/null 2>&1 || true
  echo "removed $dir ($branch); now: cd $ROOT"
}

cmd_status() {
  git -C "$ROOT" worktree list
  echo; gh pr list --state open --json number,title,isDraft,headRefName,updatedAt \
    -q '.[] | "#\(.number) \(if .isDraft then "[draft]" else "[ready]" end) \(.title)  (\(.headRefName), \(.updatedAt))"'
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  sync) shift; cmd_sync "$@" ;;
  finish) shift; cmd_finish "$@" ;;
  status) shift; cmd_status "$@" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
