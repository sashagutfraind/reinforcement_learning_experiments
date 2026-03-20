import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class BossEAEnv(gym.Env):
    """
    Boss-EA scheduling environment. No LLM calls.

    The boss has a secret favorite and disfavorite time period per day.
    Each booking receives a numeric score 0-10:
      - favorite period  → 10
      - disfavorite period → 0
      - neutral           → 5
    """

    SLOTS = 8       # 9am, 10am, ..., 4pm
    DAYS = 5        # Mon–Fri
    HOUR_OFFSET = 9 # slot 0 = 9am

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(self.DAYS * self.SLOTS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.DAYS, self.SLOTS), dtype=np.float32
        )
        self.calendar = np.zeros((self.DAYS, self.SLOTS), dtype=np.float32)
        self.boss_prefs = self._generate_boss_prefs()

    def _generate_boss_prefs(self):
        """
        For each day, randomly pick a favorite block and a disfavorite block.
        Blocks: 'morning' = slots 0-3 (9am-12pm), 'afternoon' = slots 4-7 (1pm-4pm).
        """
        prefs = {}
        for day in range(self.DAYS):
            periods = ['morning', 'afternoon']
            random.shuffle(periods)
            prefs[day] = {'favorite': periods[0], 'disfavorite': periods[1]}
        return prefs

    def _slot_period(self, slot):
        return 'morning' if slot < 4 else 'afternoon'

    def _score(self, day, slot):
        """Return boss score 0-10 for a given (day, slot) booking."""
        period = self._slot_period(slot)
        pref = self.boss_prefs[day]
        if period == pref['favorite']:
            return 10
        elif period == pref['disfavorite']:
            return 0
        return 5  # neutral (not used given only two periods, but kept for generality)

    def step(self, action):
        day = action // self.SLOTS
        slot = action % self.SLOTS
        hour = slot + self.HOUR_OFFSET

        if self.calendar[day, slot] == 1:
            # Heavy penalty for double-booking; no score from boss
            return self.calendar.copy(), -10, False, False, {'error': 'double_booking'}

        self.calendar[day, slot] = 1
        score = self._score(day, slot)

        terminated = bool(self.calendar.sum() == self.DAYS * self.SLOTS)
        info = {
            'day': day,
            'hour': hour,
            'score': score,
            'favorite_period': self.boss_prefs[day]['favorite'],
        }
        return self.calendar.copy(), score, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.calendar = np.zeros((self.DAYS, self.SLOTS), dtype=np.float32)
        # Boss preferences are set once at __init__ and never change.
        return self.calendar.copy(), {}
