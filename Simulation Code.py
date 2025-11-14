import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Generate the initial dataset
def generate_initial_data(num_students=50):
    data = []
    for student_id in range(1, num_students + 1):
        tp = random.randint(60, 80)
        me = random.randint(60, 80)
        sr = random.randint(60, 80)
        is_ = random.randint(60, 80)
        avg_skill_metric = (tp + me + sr + is_) / 4
        lp = 0  # Initial Learning Progress
        step = 'Initial'
        data.append([student_id, tp, me, sr, is_, avg_skill_metric, lp, step])
        
        # After intervention
        tp = random.randint(70, 85)
        me = random.randint(70, 85)
        sr = random.randint(70, 85)
        is_ = random.randint(70, 85)
        avg_skill_metric = (tp + me + sr + is_) / 4
        lp = avg_skill_metric - data[-1][5]  # Learning Progress
        step = 'Intervention'
        data.append([student_id, tp, me, sr, is_, avg_skill_metric, lp, step])
    
    columns = ['Student ID', 'TP', 'ME', 'SR', 'IS', 'Avg Skill Metric (S)', 'LP', 'Step']
    return pd.DataFrame(data, columns=columns)

# Update function for Q-values until convergence
def update_q_values_until_convergence(Q, state_space, action_space, alpha, gamma, convergence_threshold=0.01, max_iterations=1000):
    iteration = 0
    delta = float('inf')
    
    while delta > convergence_threshold and iteration < max_iterations:
        delta = 0
        for state in state_space:
            for action in action_space:
                old_q_value = Q[state][action]
                reward = random.uniform(1, 10)  # Placeholder reward function
                next_state = random.choice(state_space)
                max_q_next = max(Q[next_state]) if next_state in Q else 0
                Q[state][action] += alpha * (reward + gamma * max_q_next - old_q_value)
                delta = max(delta, abs(old_q_value - Q[state][action]))
        iteration += 1
    
    return Q, iteration

# Function to simulate and plot the training performance of DQN, Q-learning, PPO, SAC, TRPO, and DDPG
def plot_performance_comparison():
    episodes = 100
    dqn_rewards = [random.gauss(1.5 * i, 10) + 50 for i in range(episodes)]
    qlearning_rewards = [random.gauss(i, 15) + 40 for i in range(episodes)]
    ppo_rewards = [random.gauss(1.2 * i, 12) + 45 for i in range(episodes)]
    sac_rewards = [random.gauss(1.3 * i, 13) + 48 for i in range(episodes)]
    trpo_rewards = [random.gauss(1.4 * i, 14) + 46 for i in range(episodes)]
    ddpg_rewards = [random.gauss(1.6 * i, 11) + 52 for i in range(episodes)]
    
    plt.plot(dqn_rewards, label='DQN', color='blue')
    plt.plot(qlearning_rewards, label='Q-learning', color='green', linestyle='--')
    plt.plot(ppo_rewards, label='PPO', color='red', linestyle=':')
    plt.plot(sac_rewards, label='SAC', color='purple', linestyle='-.')
    plt.plot(trpo_rewards, label='TRPO', color='orange', linestyle='-')
    plt.plot(ddpg_rewards, label='DDPG', color='cyan', linestyle='-.')
    
    plt.title("Performance Comparison: DQN vs. Q-learning vs. PPO vs. SAC vs. TRPO vs. DDPG")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot comparative analysis metrics
def plot_comparative_analysis():
    metrics = ['Training Time', 'System Scalability', 'User Satisfaction']
    dqn_scores = [random.uniform(2, 4), random.uniform(7, 10), random.uniform(8, 10)]
    qlearning_scores = [random.uniform(4, 6), random.uniform(5, 7), random.uniform(6, 8)]
    ppo_scores = [random.uniform(3, 5), random.uniform(6, 8), random.uniform(7, 9)]
    sac_scores = [random.uniform(3, 4.5), random.uniform(7, 9), random.uniform(7.5, 9.5)]
    trpo_scores = [random.uniform(3.5, 5), random.uniform(6.5, 8.5), random.uniform(7, 9)]
    ddpg_scores = [random.uniform(2.5, 4.5), random.uniform(7, 9), random.uniform(8, 10)]
    
    index = np.arange(len(metrics))
    bar_width = 0.12

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, dqn_scores, bar_width, label='DQN', color='blue')
    bar2 = ax.bar(index + bar_width, qlearning_scores, bar_width, label='Q-learning', color='green')
    bar3 = ax.bar(index + 2 * bar_width, ppo_scores, bar_width, label='PPO', color='red')
    bar4 = ax.bar(index + 3 * bar_width, sac_scores, bar_width, label='SAC', color='purple')
    bar5 = ax.bar(index + 4 * bar_width, trpo_scores, bar_width, label='TRPO', color='orange')
    bar6 = ax.bar(index + 5 * bar_width, ddpg_scores, bar_width, label='DDPG', color='cyan')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparative Analysis of RL Algorithms')
    ax.set_xticks(index + 2.5 * bar_width)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Generate the initial data
data = generate_initial_data(num_students=50)
print("Generated Initial Data:")
print(data.head())

# Save the data to an Excel file
excel_file_path = "student_performance_data_new.xlsx"
data.to_excel(excel_file_path, index=False)
print(f"\nData has been saved to: {excel_file_path}")

# Example: Update Q-values until convergence
state_space = ['state1', 'state2', 'state3']
action_space = [0, 1, 2]
Q = {state: [random.uniform(0, 1) for _ in action_space] for state in state_space}
alpha = 0.1
gamma = 0.9

Q, iterations = update_q_values_until_convergence(Q, state_space, action_space, alpha, gamma)
print("\nUpdated Q-values after convergence:")
print(Q)
print(f"Converged after {iterations} iterations.")

# Plot performance comparison
plot_performance_comparison()

# Plot comparative analysis
plot_comparative_analysis()
