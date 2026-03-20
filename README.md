EA-Game: Reinforcement Learning Agent Experiments
================================================

This repository contains experimental code exploring reinforcement learning (RL) approaches with simple agents and environments. The goal is to prototype ideas, compare small RL setups (Q-tables, simple learning agents), and integrate language-model-driven components in some experiments.

Contents
- `rl1_qtable/` — RL for an agent acting as an Executive Assistant (Q-table RL)
- `rl2_qtable_llm/` — Refinment of RL1 : agent is now using LLM

Quickstart (using a Python virtual environment)

1. Create a virtual environment (recommended):

   python3 -m venv .venv

2. Activate the environment:

   source .venv/bin/activate

3. Install dependencies:

   - If a `requirements.txt` is present:

       pip install -r requirements.txt

   - Otherwise, install via the project packaging (pyproject.toml):

       pip install -e .

4. Run an example (adjust path to the experiment you want to try):

   python rl2_qtable_llm/main.py
   or
   python rl1_qtable/main.py

Notes
- This repo is experimental / scientific work: expect informal structure and evolving APIs.
- Use the virtual environment to avoid dependency conflicts — some users abbreviate "venv" as "uv"; the steps above assume the standard Python `venv` workflow.

Contributing
- Feel free to fork, experiment, and submit patches or notes explaining what you changed and why.

License
- MIT 
Contact
- For questions about these experiments, open an issue or contact the repository owner.
