# rl1_qtable — EA Scheduling with Tabular Bandit

A minimal reinforcement learning experiment where an **Executive Assistant (EA) agent** learns to schedule meetings at times the boss prefers, using only the boss's numeric scores as feedback.

Future work will explore more complex environments and algorithms

## Setup

**Boss** (`env.py`): Has a secret favorite half-day (morning or afternoon) per day of the week, fixed at environment creation. Scores each booking:
- Favorite period → **10**
- Disfavorite period → **0**

**Agent** (`agent.py`): Maintains a Q-table over 40 slots (5 days × 8 hours, 9am–4pm). Uses epsilon-greedy selection and a bandit update rule:

```
Q[day, slot] += alpha * (score - Q[day, slot])
```

This is a **multi-armed bandit**, not full Q-learning — there are no state transitions, so gamma is omitted. Epsilon decays each episode so the agent shifts from exploration to exploitation as it gains confidence.

## Files

| File | Description |
|------|-------------|
| `env.py` | Gymnasium environment — boss scoring logic |
| `agent.py` | Tabular bandit agent — Q-table, epsilon-greedy, decay |
| `main.py` | Training loop, progress logging, Q-table printout |
| `runs/` | CSV logs and `.npy` Q-table snapshots per run |

## Running

```bash
python main.py
```

Output includes per-episode average score (with epsilon), and a final Q-table printed as a hours × days grid:

```
Final Q-table (rows=hours, cols=days):
        Mon    Tue    Wed    Thu    Fri
9:00   10.00  10.00   0.49   1.57   1.14
10:00  10.00  10.00   0.75   0.75   1.41
...
```

Slots in the boss's favorite period converge toward 10; disfavored slots toward 0.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.1 | Learning rate — how fast Q-values update |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.95 | Multiplied each episode |
| `epsilon_min` | 0.05 | Floor on exploration |
| `NUM_EPISODES` | 100 | Training episodes |
| `BOOKINGS_PER_EPISODE` | 20 | Bookings per week |
