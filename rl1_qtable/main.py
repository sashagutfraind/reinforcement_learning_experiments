"""
Simplified EA scheduling experiment 

The boss scores each booking 0 or 10 (numeric, not verbal):
  - favorite period  → 10
  - disfavorite period → 0

The agent (EA) learns which slots the boss prefers using a tabular bandit
(epsilon-greedy + incremental Q update). Boss preferences are fixed for the
lifetime of the environment, so the agent can fully exploit what it learns.
Epsilon decays over episodes to shift from exploration toward exploitation.
"""

from env import BossEAEnv
from agent import TabularEAAgent

# --- Config ---
NUM_EPISODES = 100
BOOKINGS_PER_EPISODE = 20  # Book 20 meetings per episode

env = BossEAEnv()
agent = TabularEAAgent(alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.95)

run_history = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    episode_score = 0

    for _ in range(BOOKINGS_PER_EPISODE):
        # 1. Agent picks a slot
        action = agent.choose_action(env.calendar)
        if action is None:
            break

        # 2. Boss scores the booking (0,10)
        obs, score, terminated, truncated, info = env.step(action)

        # 3. Agent updates Q-table directly from numeric score
        agent.learn(action, score)

        day = action // env.SLOTS
        slot = action % env.SLOTS
        hour = slot + env.HOUR_OFFSET

        run_history.append({
            "episode": ep,
            "day": day,
            "hour": hour,
            "score": score,
            "favorite_period": info.get("favorite_period"),
        })
        episode_score += score

        if terminated:
            break

    agent.decay_epsilon()

    if (ep + 1) % 20 == 0:
        recent = run_history[-BOOKINGS_PER_EPISODE * 20:]
        avg = sum(r["score"] for r in recent) / len(recent)
        print(f"Episode {ep+1:4d} | avg score: {avg:.2f} | epsilon: {agent.epsilon:.3f}")

import pandas as pd

day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
hour_labels = [f"{h}:00" for h in range(env.HOUR_OFFSET, env.HOUR_OFFSET + env.SLOTS)]

# Transpose: rows = hours, cols = days
df = pd.DataFrame(agent.q_table.T, index=hour_labels, columns=day_labels)
print("\nFinal Q-table (rows=hours, cols=days):")
print(df.round(2).to_string())

agent.save_run(run_history, "rl1_qtable")
