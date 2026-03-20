import numpy as np
import os
import pandas as pd
from datetime import datetime


class TabularEAAgent:
    """
    Tabular bandit agent for scheduling.

    This is a multi-armed bandit, not full Q-learning: each (day, slot) pair
    is an independent arm. There are no meaningful state transitions between
    bookings within an episode — the boss's preferences are fixed per episode
    and revealed only through scores.

    Update rule (incremental mean / exponential moving average):
        Q[day, slot] += alpha * (reward - Q[day, slot])

    This is equivalent to Q-learning with gamma=0 (no future state value),
    which is correct here since each action's reward is immediate and final.
    """

    DAYS = 5
    SLOTS = 8
    HOUR_OFFSET = 9

    def __init__(self, alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.98):
        # Q-table: estimated expected score for each (day, slot)
        # Initialized to 5.0 (neutral / optimistic-neutral prior on 0-10 scale)
        self.q_table = np.full((self.DAYS, self.SLOTS), 5.0)
        self.alpha = alpha              # Learning rate
        self.epsilon = epsilon          # Exploration rate, decays over episodes
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, calendar):
        """Epsilon-greedy action selection over available (unbooked) slots."""
        available = np.where(calendar.flatten() == 0)[0]
        if not available.size:
            return None

        if np.random.random() < self.epsilon:
            return int(np.random.choice(available))

        q_flat = self.q_table.flatten()
        return int(available[np.argmax(q_flat[available])])

    def learn(self, action, reward):
        """
        Bandit update: move Q estimate toward observed reward.
        No gamma term — this is a stateless bandit, not sequential Q-learning.
        """
        day = action // self.SLOTS
        slot = action % self.SLOTS
        self.q_table[day, slot] += self.alpha * (reward - self.q_table[day, slot])

    def decay_epsilon(self):
        """Call once per episode. Since prefs are fixed, the agent can exploit more over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_run(self, run_data, run_id):
        os.makedirs("runs", exist_ok=True)
        df = pd.DataFrame(run_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"runs/run_{run_id}_{ts}.csv"
        df.to_csv(csv_path, index=False)
        np.save(csv_path.replace(".csv", "_qtable.npy"), self.q_table)
        print(f"Saved run to {csv_path}")
