import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Callable

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


class BossEAEnv(gym.Env):
    """
    Boss-EA scheduling environment.

    The boss has a secret favorite half-day (morning/afternoon) per day,
    fixed at environment creation. On each booking it produces a natural-language
    reaction via the provided LLM — without stating preferences explicitly.

    The true score (0 or 10) is tracked internally for evaluation but is NOT
    passed to the agent; the agent only sees the verbal response.
    """

    SLOTS = 8
    DAYS = 5
    HOUR_OFFSET = 9

    def __init__(self, llm: Callable):
        super().__init__()
        self.llm = llm
        self.action_space = spaces.Discrete(self.DAYS * self.SLOTS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.DAYS, self.SLOTS), dtype=np.float32
        )
        self.calendar = np.zeros((self.DAYS, self.SLOTS), dtype=np.float32)
        self.boss_prefs = self._generate_boss_prefs()

    def _generate_boss_prefs(self):
        prefs = {}
        for day in range(self.DAYS):
            periods = ["morning", "afternoon"]
            random.shuffle(periods)
            prefs[day] = {"favorite": periods[0], "disfavorite": periods[1]}
        return prefs

    def _slot_period(self, slot):
        return "morning" if slot < 4 else "afternoon"

    def _true_score(self, day, slot):
        period = self._slot_period(slot)
        return 10 if period == self.boss_prefs[day]["favorite"] else 0

    def _boss_response(self, day, slot):
        """Generate a brief, indirect verbal reaction via LLM."""
        hour = slot + self.HOUR_OFFSET
        period = self._slot_period(slot)
        is_favorite = period == self.boss_prefs[day]["favorite"]
        if is_favorite:
            tone = "genuinely delighted — this works perfectly for you"
        else:
            tone = "clearly frustrated — this time really does not work for you"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a busy executive. React to meeting bookings with a brief, "
                    "natural comment — one or two sentences max. Express your feelings "
                    "clearly through your tone and word choice, but do NOT state your "
                    "general preferences or say what times you prefer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Your EA just booked a meeting on {DAY_NAMES[day]} at {hour}:00. "
                    f"You feel {tone}. Write your reaction."
                ),
            },
        ]

        return self.llm(messages, max_new_tokens=60, temperature=0.8)

    def step(self, action):
        day = action // self.SLOTS
        slot = action % self.SLOTS
        hour = slot + self.HOUR_OFFSET

        if self.calendar[day, slot] == 1:
            return "That slot is already taken.", -10, False, False, {"error": "double_booking"}

        self.calendar[day, slot] = 1
        true_score = self._true_score(day, slot)
        verbal = self._boss_response(day, slot)

        terminated = bool(self.calendar.sum() == self.DAYS * self.SLOTS)
        info = {
            "day": day,
            "hour": hour,
            "true_score": true_score,
            "favorite_period": self.boss_prefs[day]["favorite"],
        }
        # Agent receives verbal; true_score is in info for logging only
        return verbal, true_score, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.calendar = np.zeros((self.DAYS, self.SLOTS), dtype=np.float32)
        # Boss preferences are fixed at __init__ and never change.
        return self.calendar.copy(), {}
