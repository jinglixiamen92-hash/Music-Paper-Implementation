import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
from typing import Dict, List, Any

class SimulationLogger:
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.episode_data = []
        self.step_data = []
        self.current_episode = 0
        self.current_step = 0
        
    def setup_logging(self, log_level):
        """Setup comprehensive logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # Root logger
        self.logger = logging.getLogger('MusicEducationSimulation')
        self.logger.setLevel(log_level)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(f'simulation_detailed_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # CSV file for structured data
        self.csv_file = f'simulation_data_{timestamp}.csv'
        self.setup_csv_logging()
        
    def setup_csv_logging(self):
        """Setup CSV file with headers"""
        headers = [
            'timestamp', 'episode', 'step', 'algorithm', 'student_id', 'student_level',
            'state_TP', 'state_ME', 'state_SR', 'state_IS', 'state_TC', 'state_EX', 'state_PC', 'state_SA',
            'action_taken', 'action_type', 'reward', 'cumulative_reward',
            'q_value', 'epsilon', 'learning_rate', 'td_error',
            'rvi', 'esi', 'training_efficiency', 'convergence_metric'
        ]
        
        with open(self.csv_file, 'w') as f:
            f.write(','.join(headers) + '\n')
    
    def log_episode_start(self, episode: int, algorithm: str, student_profile: Dict):
        """Log the start of a new episode"""
        self.current_episode = episode
        self.current_step = 0
        
        self.logger.info(f"🎵 Starting Episode {episode} - Algorithm: {algorithm}")
        self.logger.info(f"   Student Profile: {student_profile}")
        
        episode_info = {
            'episode': episode,
            'algorithm': algorithm,
            'student_profile': student_profile,
            'start_time': datetime.now(),
            'steps': [],
            'total_reward': 0,
            'final_state': None
        }
        self.episode_data.append(episode_info)
    
    def log_step(self, step_data: Dict):
        """Log detailed information for each step"""
        self.current_step += 1
        step = self.current_step
        
        # Extract data
        algorithm = step_data.get('algorithm', 'Unknown')
        student_id = step_data.get('student_id', 'Unknown')
        state = step_data.get('state', {})
        action = step_data.get('action', {})
        reward = step_data.get('reward', 0)
        q_value = step_data.get('q_value', 0)
        epsilon = step_data.get('epsilon', 0)
        
        # Create log message
        log_msg = (
            f"Step {step:03d} | Alg: {algorithm:10} | "
            f"State: TP{state.get('TP', 0):.2f} ME{state.get('ME', 0):.2f} | "
            f"Action: {action.get('type', 'Unknown'):15} | "
            f"Reward: {reward:+.2f} | Q: {q_value:.3f} | ε: {epsilon:.3f}"
        )
        
        self.logger.info(log_msg)
        
        # Save to CSV
        csv_data = {
            'timestamp': datetime.now().isoformat(),
            'episode': self.current_episode,
            'step': step,
            'algorithm': algorithm,
            'student_id': student_id,
            'student_level': step_data.get('student_level', 'Unknown'),
            'state_TP': state.get('TP', 0),
            'state_ME': state.get('ME', 0),
            'state_SR': state.get('SR', 0),
            'state_IS': state.get('IS', 0),
            'state_TC': state.get('TC', 0),
            'state_EX': state.get('EX', 0),
            'state_PC': state.get('PC', 0),
            'state_SA': state.get('SA', 0),
            'action_taken': action.get('type', 'Unknown'),
            'action_type': action.get('description', 'Unknown'),
            'reward': reward,
            'cumulative_reward': step_data.get('cumulative_reward', 0),
            'q_value': q_value,
            'epsilon': epsilon,
            'learning_rate': step_data.get('learning_rate', 0),
            'td_error': step_data.get('td_error', 0),
            'rvi': step_data.get('rvi', 0),
            'esi': step_data.get('esi', 0),
            'training_efficiency': step_data.get('training_efficiency', 0),
            'convergence_metric': step_data.get('convergence_metric', 0)
        }
        
        self.save_to_csv(csv_data)
        
        # Store in episode data
        step_record = {
            'step': step,
            'state': state,
            'action': action,
            'reward': reward,
            'q_value': q_value,
            'timestamp': datetime.now()
        }
        
        if self.episode_data:
            self.episode_data[-1]['steps'].append(step_record)
            self.episode_data[-1]['total_reward'] += reward
    
    def log_learning_update(self, update_data: Dict):
        """Log Q-learning updates and training progress"""
        loss = update_data.get('loss', 0)
        td_error = update_data.get('td_error', 0)
        learning_rate = update_data.get('learning_rate', 0)
        
        self.logger.debug(
            f"Learning Update | Loss: {loss:.4f} | TD Error: {td_error:.4f} | "
            f"LR: {learning_rate:.6f}"
        )
    
    def log_episode_end(self, episode: int, final_metrics: Dict):
        """Log the end of an episode with summary statistics"""
        duration = datetime.now() - self.episode_data[-1]['start_time']
        total_reward = self.episode_data[-1]['total_reward']
        
        self.logger.info(f"🎵 Episode {episode} Completed")
        self.logger.info(f"   Duration: {duration}")
        self.logger.info(f"   Total Reward: {total_reward:.2f}")
        self.logger.info(f"   Final Metrics: {final_metrics}")
        
        # Update episode data
        self.episode_data[-1]['end_time'] = datetime.now()
        self.episode_data[-1]['duration'] = duration
        self.episode_data[-1]['final_metrics'] = final_metrics
        self.episode_data[-1]['final_state'] = final_metrics.get('final_state', {})
    
    def log_algorithm_comparison(self, comparison_data: Dict):
        """Log comparison between different algorithms"""
        self.logger.info("=" * 80)
        self.logger.info("ALGORITHM COMPARISON RESULTS")
        self.logger.info("=" * 80)
        
        for algorithm, metrics in comparison_data.items():
            self.logger.info(
                f"{algorithm:12} | "
                f"Avg Reward: {metrics.get('avg_reward', 0):7.2f} | "
                f"RVI: {metrics.get('rvi', 0):5.3f} | "
                f"ESI: {metrics.get('esi', 0):5.3f} | "
                f"Convergence: {metrics.get('convergence', 0):5.3f}"
            )
    
    def log_student_progress(self, student_id: str, progress_data: Dict):
        """Log individual student progress over time"""
        self.logger.info(f"📊 Student {student_id} Progress Summary")
        for metric, values in progress_data.items():
            if isinstance(values, (list, tuple)) and len(values) >= 2:
                improvement = values[-1] - values[0]
                self.logger.info(f"   {metric:20}: {values[0]:.2f} → {values[-1]:.2f} "
                               f"(Δ: {improvement:+.2f})")
    
    def save_to_csv(self, data: Dict):
        """Save data to CSV file"""
        try:
            with open(self.csv_file, 'a') as f:
                row = [str(data.get(col, '')) for col in [
                    'timestamp', 'episode', 'step', 'algorithm', 'student_id', 'student_level',
                    'state_TP', 'state_ME', 'state_SR', 'state_IS', 'state_TC', 'state_EX', 'state_PC', 'state_SA',
                    'action_taken', 'action_type', 'reward', 'cumulative_reward',
                    'q_value', 'epsilon', 'learning_rate', 'td_error',
                    'rvi', 'esi', 'training_efficiency', 'convergence_metric'
                ]]
                f.write(','.join(row) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.episode_data:
            return "No simulation data available"
        
        total_episodes = len(self.episode_data)
        total_steps = sum(len(episode['steps']) for episode in self.episode_data)
        avg_reward = np.mean([episode['total_reward'] for episode in self.episode_data])
        
        report = [
            "=" * 80,
            "SIMULATION SUMMARY REPORT",
            "=" * 80,
            f"Total Episodes: {total_episodes}",
            f"Total Steps: {total_steps}",
            f"Average Reward per Episode: {avg_reward:.2f}",
            f"Simulation Duration: {self.episode_data[-1]['end_time'] - self.episode_data[0]['start_time']}",
            "",
            "Algorithm Performance:"
        ]
        
        # Group by algorithm
        algorithms = {}
        for episode in self.episode_data:
            algo = episode['algorithm']
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append(episode['total_reward'])
        
        for algo, rewards in algorithms.items():
            report.append(f"  {algo:12}: Avg Reward = {np.mean(rewards):7.2f} ± {np.std(rewards):5.2f} "
                         f"(n={len(rewards)})")
        
        report.append("=" * 80)
        
        summary_text = '\n'.join(report)
        self.logger.info("\n" + summary_text)
        
        # Save summary to file
        with open('simulation_summary.txt', 'w') as f:
            f.write(summary_text)
        
        return summary_text
    
    def export_json_logs(self, filename: str = None):
        """Export all logs to JSON format"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'simulation_export_{timestamp}.json'
        
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_episodes': len(self.episode_data),
                'total_steps': sum(len(episode['steps']) for episode in self.episode_data)
            },
            'episodes': self.episode_data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Logs exported to {filename}")
        return filename

# Example usage in your simulation
def run_simulation_with_logging():
    """Example of how to use the logger in your simulation"""
    logger = SimulationLogger(log_level=logging.INFO)
    
    # Simulate multiple episodes
    algorithms = ['DQN', 'PPO', 'Q-learning', 'SAC', 'TRPO', 'DDPG']
    
    for episode in range(10):  # Reduced for example
        algorithm = np.random.choice(algorithms)
        student_profile = {
            'student_id': f'student_{episode}',
            'level': np.random.choice(['beginner', 'intermediate', 'advanced']),
            'initial_TP': np.random.uniform(0.3, 0.8)
        }
        
        logger.log_episode_start(episode, algorithm, student_profile)
        
        # Simulate steps within episode
        cumulative_reward = 0
        for step in range(20):  # Reduced for example
            # Simulate state, action, reward
            state = {
                'TP': np.random.uniform(0, 1),
                'ME': np.random.uniform(0, 1),
                'SR': np.random.uniform(0, 1),
                'IS': np.random.uniform(0, 1),
                'TC': np.random.uniform(0, 1),
                'EX': np.random.uniform(0, 1),
                'PC': np.random.uniform(0, 1),
                'SA': np.random.uniform(0, 1)
            }
            
            action = {
                'type': np.random.choice(['TE', 'EP', 'SRP', 'CIT']),
                'description': 'Teaching strategy'
            }
            
            reward = np.random.uniform(-1, 2)
            cumulative_reward += reward
            
            step_data = {
                'algorithm': algorithm,
                'student_id': student_profile['student_id'],
                'student_level': student_profile['level'],
                'state': state,
                'action': action,
                'reward': reward,
                'cumulative_reward': cumulative_reward,
                'q_value': np.random.uniform(0, 1),
                'epsilon': max(0.1, 1.0 - episode * 0.1),
                'learning_rate': 0.001,
                'td_error': np.random.uniform(0, 0.1),
                'rvi': np.random.uniform(0.1, 0.5),
                'esi': np.random.uniform(0.5, 0.95)
            }
            
            logger.log_step(step_data)
        
        final_metrics = {
            'final_state': state,
            'total_reward': cumulative_reward,
            'learning_gain': np.random.uniform(0.1, 0.5)
        }
        
        logger.log_episode_end(episode, final_metrics)
    
    # Generate reports
    logger.generate_summary_report()
    logger.export_json_logs()
    
    return logger

# Run the example
if __name__ == "__main__":
    logger = run_simulation_with_logging()