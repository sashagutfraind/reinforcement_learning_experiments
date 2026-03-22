import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_latent_preferences(run_id=None):
    # Find the latest .npy file or a specific run_id
    search_path = f"rl2_qtable_llm/runs/run_{run_id}*.npy" if run_id else "rl2_qtable_llm/runs/*.npy"
    files = glob.glob(search_path)
    if not files:
        print("No Q-table files found in runs/")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"plot_latent_preferences: reading {latest_file}")
    q_table = np.load(latest_file)
    
    # 5 Days, 8 Hours (9am to 4pm start times)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = [f"{h}:00" for h in range(9, 17)]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(q_table.T, annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=days, yticklabels=hours)
    
    plt.title(f"EA's Learned Preference Map\n(Source: {os.path.basename(latest_file)})")
    plt.xlabel("Day of Week")
    plt.ylabel("Appointment Start Time")
    plt.tight_layout()
    plt.savefig("preference_heatmap.png")
    plt.show()

def plot_learning_curve(run_id=None):
    search_path = f"rl2_qtable_llm/runs/run_{run_id}*.csv" if run_id else "rl2_qtable_llm/runs/*.csv"
    files = glob.glob(search_path)
    if not files: return

    latest_file = max(files, key=os.path.getctime)
    print(f"plot_learning_curve: reading {latest_file}")
    df = pd.read_csv(latest_file)

    # Calculate rolling averages to see convergence
    df['rolling_perceived'] = df['perceived_score'].rolling(window=20).mean()
    df['rolling_true'] = df['true_score'].rolling(window=20).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['perceived_score'], alpha=0.2, color='gray', label='Perceived (raw)')
    plt.plot(df.index, df['rolling_perceived'], color='blue', linewidth=2, label='Perceived (rolling avg)')
    plt.plot(df.index, df['rolling_true'], color='green', linewidth=2, label='True (rolling avg)')
    
    plt.title("EA Learning Curve: Convergence of Perceived Boss Happiness")
    plt.xlabel("Total Appointments Booked")
    plt.ylabel("Reward (Sentiment Score)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_latent_preferences()
    plot_learning_curve()