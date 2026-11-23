#!/usr/bin/env bash
# The only sanctioned way to merge a PR (AGENTS.md §6). Squash-merges PR <n> into main after
# checking: targets main, open, not draft, no wip/do-not-merge label, no conflicts, CI green.
# Merging another agent's PR (branch <type>/<agent>/<topic> with agent != $AGENT_NAME) needs
# --others "<reason>"; the reason is posted on the PR, and the PR must be idle for >= 3 min.
#
#   scripts/safe_merge.sh <n> [--others "reason"] [--allow-no-ci]
# --allow-no-ci (repo has no CI workflow yet, or GitHub Actions is down): instead of trusting
# CI it runs `make check` on a local checkout of the PR head, which must equal origin.
set -euo pipefail
pr=${1:?usage: safe_merge.sh <pr-number> [--others \"reason\"] [--allow-no-ci]}; shift
others=""; allow_no_ci=0
while [ $# -gt 0 ]; do
  case $1 in
    --others) others=${2:?--others needs a reason}; shift 2 ;;
    --allow-no-ci) allow_no_ci=1; shift ;;   # no CI workflow yet / Actions down: runs make check
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
fields=number,state,isDraft,baseRefName,headRefName,mergeable,labels,updatedAt,statusCheckRollup,title
for _ in 1 2 3 4 5; do   # GitHub computes mergeability lazily
  info=$(gh pr view "$pr" --repo "$repo" --json "$fields")
  printf '%s' "$info" | grep -q '"mergeable":"UNKNOWN"' || break
  sleep 3
done
python3 - "$info" "${AGENT_NAME:-}" "$others" "$allow_no_ci" <<'PY'
import json, sys
from datetime import datetime, timezone
info, me, others, allow_no_ci = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
problems = []
if info["state"] != "OPEN": problems.append(f"PR is {info['state']}")
if info["baseRefName"] != "main": problems.append(f"base is {info['baseRefName']}, not main")
if info["isDraft"]: problems.append("PR is a draft (the author still holds the claim)")
labels = {l["name"].lower() for l in info["labels"]}
if labels & {"wip", "do-not-merge"}: problems.append(f"labelled {sorted(labels & {'wip', 'do-not-merge'})}")
if info["mergeable"] != "MERGEABLE": problems.append(f"mergeable={info['mergeable']} (run scripts/agent.sh sync and resolve)")
checks = info["statusCheckRollup"] or []
def ok(c):
    if c.get("__typename") == "StatusContext":
        return c.get("state") == "SUCCESS"
    return c.get("status") == "COMPLETED" and c.get("conclusion") in ("SUCCESS", "SKIPPED", "NEUTRAL")
bad = [c.get("name") or c.get("context") for c in checks if not ok(c)]
if bad: problems.append(f"CI not green: {bad}")
if not checks and not allow_no_ci: problems.append("no CI checks reported yet; wait for CI")
parts = info["headRefName"].split("/")
owner = parts[1] if len(parts) >= 3 else "?"
if owner != me and not others:
    problems.append(f"branch belongs to agent '{owner}', you are '{me or '$AGENT_NAME unset'}': "
                    "use --others \"reason\" only per AGENTS.md §6.3")
if others:
    idle = (datetime.now(timezone.utc) - datetime.fromisoformat(info["updatedAt"].replace("Z", "+00:00"))).total_seconds()
    if idle < 180: problems.append(f"PR updated {idle:.0f}s ago; the author may still be working (wait >= 3 min)")
if problems:
    print(f"REFUSING to merge #{info['number']} ({info['title']}):", *problems, sep="\n  - ", file=sys.stderr)
    sys.exit(1)
print(f"#{info['number']} passes all merge checks")
PY
if [ "$allow_no_ci" = 1 ]; then
  head=$(printf '%s' "$info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["headRefName"])')
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$head" ] || { echo "--allow-no-ci: run from a checkout of $head (local make check replaces CI)" >&2; exit 1; }
  git fetch -q origin "$head"
  [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$head")" ] || { echo "--allow-no-ci: local $head differs from origin; push or pull first" >&2; exit 1; }
  make check || { echo "--allow-no-ci: make check failed; not merging" >&2; exit 1; }
fi
if [ -n "$others" ]; then
  gh pr comment "$pr" --repo "$repo" --body "Merged by \`${AGENT_NAME:-unknown}\` (AGENTS.md §6.3): $others"
fi
gh pr merge "$pr" --repo "$repo" --squash --delete-branch
echo "merged #$pr. Everyone: scripts/agent.sh sync"
