# rl2_qtable_llm — EA Scheduling with LLM-mediated Feedback

Extends `rl1_qtable` by replacing the direct numeric reward signal with
two LLM calls run locally via the HuggingFace `transformers` pipeline:

1. **Boss** — generates a brief, verbal reaction to each booking
2. **Agent** — interprets that reaction as a score (0–10) to use for learning

Everything else (Q-table structure, bandit update, epsilon-greedy policy,
epsilon decay) is identical to `rl1_qtable`.

## Setup

```bash
uv sync
```

No API key required — the model runs locally (~1 GB download on first use).

## How it works

**Boss** (`env.py`): Has a secret favorite half-day per day (fixed at init).
Instead of returning a number, it prompts an LLM to react naturally
to the booking — e.g. *"Friday at 3pm? That's absolutely perfect, thank you!"*
or *"Monday morning?! That really doesn't work for me at all."*
The boss expresses feelings clearly but never states its general preferences.

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
| `llm.py` | LLM loader — wraps transformers pipeline as a callable |
| `env.py` | Gymnasium env — LLM-powered boss reactions |
| `agent.py` | Tabular bandit agent — LLM score extraction + Q-update |
| `main.py` | Training loop — tqdm progress bar, colored Q-table output |
| `runs/` | CSV logs (include `boss_response`, `perceived_score`, `true_score`) |

## Running

```bash
python main.py
```

Progress is shown via a tqdm bar with live stats:

```
Episodes: 42%|████▏     | 42/100 [12:31<17:15, true=7.20, perceived=6.80, eps=0.123]
```

The final Q-table is printed with color highlighting (green = favored, red = disfavored)
and a legend.

## What this tests

The key question is whether a small local model (0.5B) can reliably read sentiment
from natural language — and whether that noisy perceived signal is still clean enough
to drive learning. Compare the final Q-table and convergence speed against `rl1_qtable`
to measure the cost of LLM-mediated reward.

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

More effective RL, allowing the agent to make intelligent guesses from cell to cell
in the calendar, rather than seeing each cell in isolation.
