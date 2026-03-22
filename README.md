
Glyptic RL: Emergent World Models in LLM-powered Agents
================================================

This research project introduces Glyptic Knowledge, a framework for Reinforcement Learning (RL) designed to bridge the "Pragmatic Gap" in LLM-powered agents. While standard Large Language Models excel as "Spectators"—possessing vast amounts of explicit, pre-trained knowledge—they often struggle to adapt to the "friction" of real-world settings where documentation is incomplete or deceptive. By implementing a Glyptic Layer, we demonstrate how agents can transition from probabilistic guessing to Mastery, in a process similar to Groking in LLM training. This process, rooted in Deweyan pragmatism, allows agents to learnt their mission, eventually triggering discrete epiphanies where the agent recognizes and articulates hidden environmental rules that contradict its initial training priors.

Our initial simulations, focused on a "Disfavored Calendar" toy system: an executive assistant (EA) is trying to learn the best time to schedule tasks. Additional results, including expanded case studies in adversarial negotiation and legacy system archeology, are currently in development and will be published to this repository in the coming months.


## Contents
- `rl1_qtable/` — RL for an agent acting as an Executive Assistant (Q-table RL)
- `rl2_qtable_llm/` — Refinment of RL1 : agent is now using LLM

## Quickstart

1. Install dependencies:

   uv sync

2. Run an experiment:

   python rl2_qtable_llm/main.py
   or
   python rl1_qtable/main.py

## Notes
- This repo is experimental / scientific work: expect informal structure and evolving APIs.
- Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml`.

## Contributing
- Feel free to fork, experiment, and submit patches or notes explaining what you changed and why.

## License
- MIT

## Contact
- For questions about these experiments, open an issue or contact the repository owner.
