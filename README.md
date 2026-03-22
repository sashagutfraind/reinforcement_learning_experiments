EA-Game: Reinforcement Learning Agent Experiments
================================================

This repository contains experimental code exploring reinforcement learning (RL) approaches with simple agents and environments. The goal is to prototype ideas, compare small RL setups (Q-tables, simple learning agents), and integrate language-model-driven components in some experiments.

Contents
- `rl1_qtable/` — RL for an agent acting as an Executive Assistant (Q-table RL)
- `rl2_qtable_llm/` — Refinment of RL1 : agent is now using LLM

Quickstart

1. Install dependencies:

   uv sync

2. Run an experiment:

   python rl2_qtable_llm/main.py
   or
   python rl1_qtable/main.py

Notes
- This repo is experimental / scientific work: expect informal structure and evolving APIs.
- Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml`.

Contributing
- Feel free to fork, experiment, and submit patches or notes explaining what you changed and why.

License
- MIT

Contact
- For questions about these experiments, open an issue or contact the repository owner.
