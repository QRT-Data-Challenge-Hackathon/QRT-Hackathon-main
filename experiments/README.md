# experiments/

One folder per agent: `experiments/<AGENT_NAME>/`. Only that agent edits it, so parallel agents
never conflict here. Start from `notebooks/02_experiment_template.ipynb`, one idea per notebook
(`NN_<idea>.ipynb`), and save every result with `tracking.save_run` so anyone can blend it.
Promote code that works into `src/qrtc/` via a small PR with tests. Commit notebooks without
outputs (`make nbstrip`).
