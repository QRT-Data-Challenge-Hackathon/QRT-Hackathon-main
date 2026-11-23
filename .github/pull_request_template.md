## What
<!-- one or two sentences; one idea per PR -->

## Agent / scope
- Agent: `$AGENT_NAME`
- Files touched:

## Evidence
- Run id(s): `…`  CV (metric): `mean ± std`  vs baseline `run_id`: `mean ± std`
- [ ] Gain > ~1 fold-std (else: code only, no claim) · [ ] leak check done for big gains

## Checklist (AGENTS.md)
- [ ] `make check` green locally
- [ ] synced with `origin/main` (`scripts/agent.sh sync`), no conflicts
- [ ] no contract files changed (competition.toml, cv/metrics/submit/io/config) unless I am the Validator
- [ ] no data, secrets or notebook outputs committed
