"""
rl2_qtable_llm — EA Scheduling with LLM-mediated feedback

The boss generates a natural-language reaction via Phi-3.5-mini-instruct,
without revealing its preferences directly. The agent uses Phi-3.5-mini-instruct
to interpret that reaction as a score (0-10), then learns from it.

The model is loaded once and passed as a callable to both Env and Agent.
"""

import pandas as pd
from tqdm import tqdm
from llm import make_llm
from env import BossEAEnv
from agent import TabularEAAgent

# --- Config ---
NUM_EPISODES = 100
BOOKINGS_PER_EPISODE = 20

# Load model once; both env and agent share the same callable
llm = make_llm()

env = BossEAEnv(llm=llm)
agent = TabularEAAgent(llm=llm, alpha=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.95)

run_history = []

pbar = tqdm(range(NUM_EPISODES), desc="Episodes", unit="ep")
for ep in pbar:
    obs, info = env.reset()
    episode_true_score = 0

    for _ in range(BOOKINGS_PER_EPISODE):
        # 1. Agent picks a slot
        action = agent.choose_action(env.calendar)
        if action is None:
            break

        # 2. Boss responds verbally; true_score tracked in info for evaluation only
        verbal, true_score, terminated, truncated, info = env.step(action)

        # 3. Agent interprets the verbal response as a perceived score
        perceived_score = agent.extract_score(verbal)

        # 4. Agent learns from perceived score (not true score)
        agent.learn(action, perceived_score)

        day = action // env.SLOTS
        slot = action % env.SLOTS
        hour = slot + env.HOUR_OFFSET

        run_history.append({
            "episode": ep,
            "day": day,
            "hour": hour,
            "boss_response": verbal,
            "perceived_score": perceived_score,
            "true_score": true_score,
            "favorite_period": info.get("favorite_period"),
        })
        episode_true_score += true_score

        if terminated:
            break

    agent.decay_epsilon()

    if run_history:
        recent = run_history[-BOOKINGS_PER_EPISODE * 5:]
        avg_true = sum(r["true_score"] for r in recent) / len(recent)
        avg_perceived = sum(r["perceived_score"] for r in recent) / len(recent)
        pbar.set_postfix(
            true=f"{avg_true:.2f}",
            perceived=f"{avg_perceived:.2f}",
            eps=f"{agent.epsilon:.3f}",
        )

day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
hour_labels = [f"{h}:00" for h in range(env.HOUR_OFFSET, env.HOUR_OFFSET + env.SLOTS)]

df = pd.DataFrame(agent.q_table.T, index=hour_labels, columns=day_labels)

# ANSI color codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"

def _cell_color(val, vmin, vmax):
    """Map a Q-value to an ANSI color based on its position in [vmin, vmax]."""
    span = vmax - vmin if vmax > vmin else 1.0
    t = (val - vmin) / span  # 0.0 = worst, 1.0 = best
    if t >= 0.75:
        return _BOLD + _GREEN   # strongly favored
    elif t >= 0.5:
        return _CYAN            # mildly favored
    elif t >= 0.25:
        return _YELLOW          # mildly disfavored
    else:
        return _RED             # strongly disfavored

vmin = agent.q_table.min()
vmax = agent.q_table.max()

col_width = 8
header = " " * 6 + "".join(f"{d:>{col_width}}" for d in day_labels)
print("\nFinal Q-table (rows=hours, cols=days):")
print(header)
for hour in hour_labels:
    row = f"{hour:<6}"
    for day in day_labels:
        val = df.loc[hour, day]
        color = _cell_color(val, vmin, vmax)
        row += color + f"{val:>{col_width}.2f}" + _RESET
    print(row)

print()
print("Legend:")
print(f"  {_BOLD}{_GREEN}■{_RESET} strongly favored  (top 25%)")
print(f"  {_CYAN}■{_RESET} mildly favored    (50–75%)")
print(f"  {_YELLOW}■{_RESET} mildly disfavored (25–50%)")
print(f"  {_RED}■{_RESET} strongly disfavored (bottom 25%)")

agent.save_run(run_history, "rl2_qtable_llm")
