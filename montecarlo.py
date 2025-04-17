import numpy as np
from collections import defaultdict
import random
import pickle

# Base Agent
class MonteCarloAgent:
    def __init__(self, env, epsilon=0.01, gamma=0.99):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = defaultdict(lambda: np.zeros(env.dim[1]))
        self.state_action_counts = defaultdict(lambda: np.zeros(env.dim[1]))
        self.episode_count = 0
    
    # Convert state to a hashable key
    def state_to_key(self, state):
        return tuple(state)
    
    # e-greedy policy
    def policy(self, state, greedy=False):
        valid_actions = self.env.valid_actions()
        
        if not valid_actions:
            return None
            
        state_key = self.state_to_key(state)
        
        if random.random() > self.epsilon or greedy:
            # Greedy action selection
            q_values = self.q_values[state_key]
            
            # Filter for valid actions
            valid_q_values = {a: q_values[a] for a in valid_actions}
            if not valid_q_values:
                return random.choice(valid_actions)
                
            # random tie-breaking for max action
            max_q = max(valid_q_values.values())
            best_actions = [a for a, q in valid_q_values.items() if q == max_q]
            return random.choice(best_actions)
        else:
            # Random action
            return random.choice(valid_actions)
    
    def train(self, num_episodes=1000):
        raise NotImplementedError("Subclasses to implement this method")

# On Policy MC
class OnPolicyMC(MonteCarloAgent):
    def __init__(self, env, epsilon=0.1, gamma=0.99):
        super().__init__(env, epsilon, gamma)
        
    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            # Generate an episode
            states, actions, rewards = self.env.simulate_episode(
                agent_policy_fn=lambda x: self.policy(x)
            )
            
            # Calculate returns
            G = 0
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + self.gamma * G
                
                state_key = self.state_to_key(states[t])
                action = actions[t]
                
                # Update state-action counts
                self.state_action_counts[state_key][action] += 1
                
                # Update Q-value
                count = self.state_action_counts[state_key][action]
                self.q_values[state_key][action] += (G - self.q_values[state_key][action]) / count
            
            self.episode_count += 1

# Off Policy MC
class OffPolicyMC(MonteCarloAgent):
    def __init__(self, env, epsilon=0.1, gamma=0.99):
        super().__init__(env, epsilon, gamma)
        self.c_values = defaultdict(lambda: np.zeros(env.dim[1]))
        self.behavior_epsilon = 0.25
        
    def behavior_policy(self, state):
        valid_actions = self.env.valid_actions()
        
        if not valid_actions:
            return None
        
        if random.random() > self.behavior_epsilon:
            state_key = self.state_to_key(state)
            q_values = self.q_values[state_key]
            
            # Filter for valid actions
            valid_q_values = {a: q_values[a] for a in valid_actions}
            if not valid_q_values:
                return random.choice(valid_actions)
            
            # random tie-breaking for max action
            max_q = max(valid_q_values.values())
            best_actions = [a for a, q in valid_q_values.items() if q == max_q]
            return random.choice(best_actions)
        else:
            return random.choice(valid_actions)
    
    def target_policy(self, state):
        return self.policy(state, True)
    
    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            # Generate an episode 
            states, actions, rewards = self.env.simulate_episode(
                agent_policy_fn=lambda s: self.behavior_policy(s)
            )
            
            G = 0
            W = 1.0 
            
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + self.gamma * G
                
                state_key = self.state_to_key(states[t])
                action = actions[t]
                
                # Update cumulative weight
                self.c_values[state_key][action] += W
                
                # Update Q-value using weighted importance sampling
                if self.c_values[state_key][action] > 0:
                    self.q_values[state_key][action] += (W / self.c_values[state_key][action]) * (G - self.q_values[state_key][action])
                
                # If the action taken by the behavior policy is not the target action, then exit the loop
                if action != self.target_policy(states[t]):
                    break
                    
                valid_actions = self.env.valid_actions()
                n_valid = len(valid_actions) if valid_actions else 1
                
                # Probability of taking action under behavior policy
                prob_behavior = (1 - self.behavior_epsilon) if action == self.behavior_policy(states[t]) else self.behavior_epsilon / n_valid
                prob_target = 1.0
                
                # Update weight with importance sampling ratio
                if prob_behavior > 0:
                    W *= prob_target / prob_behavior
            
            self.episode_count += 1