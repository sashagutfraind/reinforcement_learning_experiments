import re
import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Callable


class TabularEAAgent:
    """
    Tabular bandit agent for scheduling.

    Same bandit Q-update as rl1, but reward is no longer a direct number:
    the agent uses the provided LLM to interpret the boss's verbal response
    as a perceived score (0-10), which it then uses for learning.

    Update rule:
        Q[day, slot] += alpha * (perceived_score - Q[day, slot])
    """

    DAYS = 5
    SLOTS = 8
    HOUR_OFFSET = 9

    def __init__(self, llm: Callable, alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.95):
        self.llm = llm
        self.q_table = np.full((self.DAYS, self.SLOTS), 5.0)
        self.alpha = alpha
        self.epsilon = epsilon
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

    def extract_score(self, boss_text: str) -> float:
        """
        Interpret the boss's verbal reaction as a score from 0 to 10.
        Falls back to 5.0 (neutral) if parsing fails.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You rate how happy a boss sounds about a meeting being booked. "
                    "Reply with a single integer from 0 (very unhappy) to 10 (very happy). "
                    "No explanation, just the number."
                ),
            },
            {
                "role": "user",
                "content": f'Boss said: "{boss_text}"',
            },
        ]

        try:
            text = self.llm(messages, max_new_tokens=5, temperature=0.0)
            match = re.search(r"\d+", text)
            if match:
                return float(np.clip(int(match.group()), 0, 10))
        except Exception:
            pass
        return 5.0  # neutral fallback

    def learn(self, action, perceived_score):
        """Bandit update toward perceived score."""
        day = action // self.SLOTS
        slot = action % self.SLOTS
        self.q_table[day, slot] += self.alpha * (perceived_score - self.q_table[day, slot])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_run(self, run_data, run_id):
        os.makedirs("runs", exist_ok=True)
        df = pd.DataFrame(run_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"runs/run_{run_id}_{ts}.csv"
        df.to_csv(csv_path, index=False)
        np.save(csv_path.replace(".csv", "_qtable.npy"), self.q_table)
        print(f"Saved run to {csv_path}")
