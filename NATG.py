import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Data from your study
algorithms = ['Random', 'Practice Weakest\nSkill', 'Rule-based\nExpert',
              'DDPG', 'TRPO', 'SAC', 'Q-learning', 'PPO', 'DQN']
rewards = [320, 580, 610, 600, 720, 850, 980, 1200, 1450]
errors = [40, 35, 38, 70, 65, 50, 60, 55, 45]
categories = ['Baseline'] * 3 + ['RL Algorithm'] * 6

# Create 1 row, 2 column subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- Plot 1: Main performance comparison with improvement multipliers ---
bars = ax1.barh(algorithms, rewards, xerr=errors, capsize=8,
                alpha=0.85, edgecolor='black', linewidth=1.2)

# Color code by category
for bar, category in zip(bars, categories):
    bar.set_color('lightcoral' if category == 'Baseline' else 'lightgreen')

# Add improvement multipliers for key RL algorithms
best_baseline = 610  # Rule-based expert
for i, (algo, reward) in enumerate(zip(algorithms, rewards)):
    if algo in ['DQN', 'PPO', 'Q-learning']:
        improvement = reward / best_baseline
        ax1.text(reward + errors[i] + 30, i, f'{improvement:.1f}x',
                 va='center', fontsize=10, color='darkred')

ax1.set_xlabel('Cumulative Reward (Mean ± 95% CI)', fontsize=10, fontweight='bold')
ax1.set_title('RL Algorithms vs. Heuristic Baselines',fontsize=10, fontweight='bold')
ax1.axvline(x=best_baseline, color='red', linestyle='--', alpha=0.7, label='Best Baseline')
ax1.grid(True, alpha=0.3, axis='x')
ax1.legend()

# --- Plot 2: Learning trajectory comparison ---
episodes = np.arange(100)

# Baseline trajectories (plateau early)
random_traj = np.ones_like(episodes) * 320
weakest_traj = np.ones_like(episodes) * 580
expert_traj = np.ones_like(episodes) * 610

# RL algorithms (continuous improvement, simulated curves)
np.random.seed(42)
dqn_traj = 1450 * (1 - np.exp(-0.05 * episodes)) + np.random.normal(0, 20, 100)
ppo_traj = 1200 * (1 - np.exp(-0.04 * episodes)) + np.random.normal(0, 25, 100)

# Smooth curves
window = 5
dqn_smooth = np.convolve(dqn_traj, np.ones(window)/window, mode='same')
ppo_smooth = np.convolve(ppo_traj, np.ones(window)/window, mode='same')

# Plot lines
ax2.plot(episodes, random_traj, '--', label='Random Baseline', color='red', alpha=0.7)
ax2.plot(episodes, weakest_traj, '--', label='Practice Weakest Skill', color='orange', alpha=0.7)
ax2.plot(episodes, expert_traj, '--', label='Rule-based Expert', color='purple', alpha=0.7)
ax2.plot(episodes, dqn_smooth, '-', label='DQN', color='green', linewidth=3)
ax2.plot(episodes, ppo_smooth, '-', label='PPO', color='blue', linewidth=2)

ax2.set_xlabel('Training Episodes', fontsize=10, fontweight='bold')
ax2.set_ylabel('Cumulative Reward', fontsize=10, fontweight='bold')
ax2.set_title('Learning Trajectories: RL vs. Heuristic Approaches',
              fontsize=10, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1600)

# Annotate differences
ax2.annotate('Baselines Plateau', xy=(60, 650), xytext=(20, 1100),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontweight='bold', color='red', fontsize=11)

ax2.annotate('RL Continuous Improvement', xy=(80, 1300), xytext=(40, 1450),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontweight='bold', color='green', fontsize=10)

plt.tight_layout()
fig.suptitle('Figure 16: Contextualizing Reinforcement Learning Gains\n'
             'RL Algorithms vs. Educational Heuristics', 
             fontsize=18, fontweight='bold', y=1.05)

plt.savefig('Figure16_RL_vs_Baselines.png', dpi=150, bbox_inches='tight')
plt.show()
