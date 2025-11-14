import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Parameters
n_algorithms = 6
n_seeds = 10
algorithms = ['DQN', 'Q-learning', 'PPO', 'SAC', 'TRPO', 'DDPG']

# Base reward-to-penalty ratios
base_ratios = {
    'DQN': 5.00,
    'Q-learning': 2.33, 
    'PPO': 4.50,
    'SAC': 3.33,
    'TRPO': 1.50,
    'DDPG': 1.39
}

# Generate synthetic data
np.random.seed(42)
def generate_ratio_data(algorithm_name, n_seeds=10):
    base_ratio = base_ratios[algorithm_name]
    if algorithm_name == 'DQN':
        variation = np.random.normal(0, 0.15, n_seeds)
    elif algorithm_name == 'PPO':
        variation = np.random.normal(0, 0.25, n_seeds)
    elif algorithm_name == 'SAC':
        variation = np.random.normal(0, 0.30, n_seeds)
    elif algorithm_name == 'Q-learning':
        variation = np.random.normal(0, 0.35, n_seeds)
    elif algorithm_name == 'TRPO':
        variation = np.random.normal(0, 0.20, n_seeds)
    else:  # DDPG
        variation = np.random.normal(0, 0.18, n_seeds)
    ratios = base_ratio + variation
    return np.maximum(ratios, 0.5)

ratio_data = {algo: generate_ratio_data(algo, n_seeds) for algo in algorithms}

# Calculate mean & CI
means, cis = [], []
for algo in algorithms:
    data = ratio_data[algo]
    means.append(np.mean(data))
    cis.append(1.96 * stats.sem(data))

# Create side-by-side figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ---------------- Subplot (a): Reward-to-Penalty Ratio ----------------
bars = ax1.bar(algorithms, means, yerr=cis, capsize=8, alpha=0.8,
               edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Algorithms', fontsize=13, fontweight='bold')
ax1.set_ylabel('Reward-to-Penalty Ratio', fontsize=13, fontweight='bold')
ax1.set_title('(a) Mean Ratio ± 95% CI', fontsize=14, fontweight='bold')

# Add labels and qualitative interpretation
for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + ci + 0.1,
             f'{mean:.2f} ± {ci:.2f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
    
    if mean >= 4.0:
        interp, color = "Excellent", "green"
    elif mean >= 2.5:
        interp, color = "Good", "blue"
    elif mean >= 1.5:
        interp, color = "Moderate", "orange"
    else:
        interp, color = "Poor", "red"
    ax1.text(bar.get_x() + bar.get_width()/2., -0.3, interp,
             ha='center', va='top', fontsize=9, fontweight='bold', color=color)

ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, axis='y', alpha=0.3)

# Significance stars vs DQN
dqn_ratios = ratio_data['DQN']
for i, algo in enumerate(algorithms[1:], 1):
    stat, p = stats.mannwhitneyu(dqn_ratios, ratio_data[algo], alternative='greater')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax1.text(i, means[i] + cis[i] + 0.2, sig,
             ha='center', va='bottom', fontsize=14, color='red', fontweight='bold')

ax1.text(0.02, 0.97,
         "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Threshold lines
ax1.axhline(y=4.0, color='green', linestyle='--', alpha=0.3)
ax1.axhline(y=2.5, color='blue', linestyle='--', alpha=0.3)
ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.3)
ax1.set_ylim(-0.5, max(means) + max(cis) + 0.6)

# ---------------- Subplot (b): Relative Ratios ----------------
dqn_mean = means[0]
rel_means, rel_cis = [], []
for algo in algorithms:
    rel = ratio_data[algo] / dqn_mean
    rel_means.append(np.mean(rel))
    rel_cis.append(1.96 * stats.sem(rel))

bars2 = ax2.bar(algorithms, rel_means, yerr=rel_cis, capsize=6,
                alpha=0.7, edgecolor='black', color='purple')

ax2.set_ylabel('Relative to DQN', fontsize=13, fontweight='bold')
ax2.set_title('(b) Relative Performance (DQN=1.0)', fontsize=14, fontweight='bold')
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='DQN baseline')
ax2.legend()

for bar, ratio in zip(bars2, rel_means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
             f'{ratio:.2f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, axis='y', alpha=0.3)

# ---------------- Layout ----------------
plt.tight_layout()
plt.savefig('Figure_13_RewardPenalty_TwoSubplots.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_13_RewardPenalty_TwoSubplots.pdf', bbox_inches='tight')
plt.show()
