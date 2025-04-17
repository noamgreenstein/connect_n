import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import os
from connectn import ConnectN
from montecarlo import OnPolicyMC, OffPolicyMC
from dqn import DQNAgent

class ExperimentRunner:
    def __init__(self):
        self.n = [n for n in range(3, 7)]
        self.agent_types = ["On-Policy MC", "Off-Policy MC", "DQN"]
        
    def train_agents(self, n, num_episodes=2000, eval_interval=100):
        env = ConnectN(n)

        dqn_params = {
            'observation_dim': env.dim[0] * env.dim[1],
            'action_dim': env.dim[1],
            'action_space': [_ for _ in range(env.dim[1])],
            'hidden_layer_dim': 128,
            'gamma': 0.99,
            'epsilon_start_value': 1.0,
            'epsilon_end_value': 0.1, 
            'epsilon_duration': 10_000,  
            'replay_buffer_size': 10000, 
            'freq_update_behavior_policy': 1, 
            'freq_update_target_policy': 50, 

            'batch_size': 32,
            'learning_rate': 1e-3, 
            'model_name': f'connect_{n}'
        }
        
        on_policy_mc = OnPolicyMC(env, epsilon=0.1, gamma=0.99)
        off_policy_mc = OffPolicyMC(env, epsilon=0.1, gamma=0.99)
        dqn_agent = DQNAgent(env, dqn_params)
        
        agents = [on_policy_mc, off_policy_mc, dqn_agent]
        stats = {name: {"win_rates": [], "training_times": []} for name in self.agent_types}
        
        for agent, name in zip(agents, self.agent_types):
            print(f"\nTraining {name} on Connect-{n}")
            total_training_time = 0
            
            for episode in range(0, num_episodes, eval_interval):
                episode_start_time = time.time()
                
                # Print progress
                print(f"Episodes {episode}-{episode+eval_interval-1}")
                
                # Train agent
                agent.train(num_episodes=eval_interval)
                
                episode_end_time = time.time()
                
                training_time = episode_end_time - episode_start_time
                total_training_time += training_time
                stats[name]["training_times"].append(total_training_time)
                
                win_rate = self.evaluate_agent(agent, env, num_games=100)
                stats[name]["win_rates"].append(win_rate)
                print(f" Episode {episode+eval_interval}: Win Rate = {win_rate:.2f}, Time = {training_time:.2f}s (Total: {total_training_time:.2f}s)")

        return agents, stats
    
    def evaluate_agent(self, agent, env, num_games=100):
        wins = 0
        
        for _ in range(num_games):
            state = env.reset()
            done = False
            
            while not done:
                if env.current_player == 1:
                    action = agent.policy(state, True)
                else: 
                    action = env.random_action()
                
                state, reward, done = env.step(action)
                
                if done and reward == 1:
                    wins += 1
        
        return wins / num_games * 100
    
    def run_experiments(self, num_episodes=100000, eval_interval=100, trial_num=1):
        results = {}
        
        for cn in self.n:
            print(f"\nRunning experiments for Connect-{cn}")
            agents, stats = self.train_agents(cn, num_episodes, eval_interval)
            results[cn] = stats
            
            self.plot_results(cn, stats, eval_interval, trial_num)
            df = pd.DataFrame()
            for a in self.agent_types:
                temp = pd.DataFrame(stats[a])
                temp['Agent'] = [a for _ in range(temp.shape[0])]
                df = pd.concat([df, temp], axis=0)
            df.to_csv(f'results/c{cn}_trial{trial_num}.csv')
        
        return results
    
    # Plot Win Rate and Training Times
    def plot_results(self, n, stats, eval_interval, trial_num):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for agent_name in self.agent_types:
            win_rates = stats[agent_name]["win_rates"]
            episodes = np.arange(eval_interval, eval_interval * (len(win_rates) + 1), eval_interval)
            ax1.plot(episodes, win_rates, label=agent_name)
            
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_title(f"Win Rate vs Episodes for Connect-{n}")
        ax1.legend()
        
        for agent_name in self.agent_types:
            training_times = stats[agent_name]["training_times"]
            episodes = np.arange(eval_interval, eval_interval * (len(training_times) + 1), eval_interval)
            ax2.plot(episodes, training_times, label=agent_name)
            
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Cumulative Training Time")
        ax2.set_title(f"Training Time vs Episodes for Connect-{n}")
        ax2.legend()
        ax2.grid(True)
        plt.savefig(f'results/connect_{n}_trial{trial_num}.png')
