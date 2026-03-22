# rl2_qtable_llm — EA Scheduling with LLM-mediated Feedback

Extends `rl1_qtable` by replacing the direct numeric reward signal with
two LLM calls using **microsoft/Phi-3.5-mini-instruct** via the HuggingFace
Inference API:

1. **Boss** — generates a brief, verbal reaction to each booking
2. **Agent** — interprets that reaction as a score (0–10) to use for learning

Everything else (Q-table structure, bandit update, epsilon-greedy policy,
epsilon decay) is identical to `rl1_qtable`.

Future work will explore more complex environments and algorithms.

## Setup

```bash
export HF_TOKEN=your_token_here
uv sync   # installs huggingface-hub
```

## How it works

**Boss** (`env.py`): Has a secret favorite half-day per day (fixed at init).
Instead of returning a number, it prompts an LLM to react naturally
to the booking without stating its preference outright — e.g. *"This works,
though I do find early afternoons a bit disruptive."*

**Agent** (`agent.py`): Passes the boss's verbal response back to
the LLM and asks it to rate the boss's apparent happiness on 0–10.
That perceived score drives the Q-table update:

```
Q[day, slot] += alpha * (perceived_score - Q[day, slot])
```

The true score (0 or 10) is logged for evaluation but **never seen by the agent**.

## Files

| File | Description |
|------|-------------|
| `env.py` | Gymnasium env — LLM-powered boss reactions |
| `agent.py` | Tabular bandit agent — LLM score extraction + Q-update |
| `main.py` | Training loop — logs both perceived and true scores |
| `runs/` | CSV logs (include `boss_response`, `perceived_score`, `true_score`) |

## Running

```bash
python main.py
```

Progress output compares true and perceived average scores so you can see
how well the agent's LLM interpreter tracks the actual signal:

```
Episode   20 | avg true: 6.50 | avg perceived: 6.10 | epsilon: 0.358
Episode   40 | avg true: 8.00 | avg perceived: 7.80 | epsilon: 0.129
...
```

## What this tests

The key question is whether Phi-3.5-mini can reliably read sentiment from
deliberately indirect language — and whether that noisy perceived signal is
still clean enough to drive learning. Compare the final Q-table and
convergence speed against `rl1_qtable` to measure the cost of LLM-mediated
reward.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.1 | Learning rate — how fast Q-values update |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.95 | Multiplied each episode |
| `epsilon_min` | 0.05 | Floor on exploration |
| `NUM_EPISODES` | 100 | Training episodes |
| `BOOKINGS_PER_EPISODE` | 20 | Bookings per episode |


## Future
More effective RL, allowing the agent to make intelligent guesses from cell to cell in the calendar, rather than seeing each cell in isolation.