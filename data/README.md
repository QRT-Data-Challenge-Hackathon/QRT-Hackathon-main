# data/

Gitignored (except this file). Organiser files go in `data/raw/`; rehearsal data is generated
into `data/synthetic/<kind>/` by `make synth KIND=<kind>`.

Always in the **main checkout** (not in agent worktrees): `qrtc.config.SHARED_ROOT` resolves
relative data paths there, so every worktree reads the same files. Never commit data.
