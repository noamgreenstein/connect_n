import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Three Layer DQN
class DeepQNet(nn.Module):
    def __init__(self, input_dim, dim_hidden_layer, output_dim):
        super(DeepQNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = dim_hidden_layer
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden(x))
        y = self.output_layer(x)
        return y

# Replay Memory
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.total_size = buffer_size
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        trans = (obs, act, reward, next_obs, done)
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []

        for idx in indices:
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.asarray(act))
            rewards_list.append(np.asarray(reward))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.asarray(d))

        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sample_batch(self, batch_size):
        # Make sure we have enough samples in the buffer
        if len(self._data_buffer) < batch_size:
            batch_size = len(self._data_buffer)
            
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)

# Linear Esilon Schedule
class LinearSchedule(object):
    def __init__(self, start_value, end_value, duration):
        self._start_value = start_value
        self._end_value = end_value
        self._duration = duration
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        return self._start_value + self._schedule_amount * min(1.0, time * 1.0 / self._duration)

# Initial Weights for NN
def customized_weights_init(m):
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)

# DQN Agent
class DQNAgent(object):
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params['action_dim']
        self.obs_dim = params['observation_dim']
        self.action_space = params['action_space']
        self.behavior_policy_net = DeepQNet(input_dim=params['observation_dim'],
                                   dim_hidden_layer=params['hidden_layer_dim'],
                                   output_dim=params['action_dim'])
        self.target_policy_net = DeepQNet(input_dim=params['observation_dim'],
                                          dim_hidden_layer=params['hidden_layer_dim'],
                                          output_dim=params['action_dim'])
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=params['learning_rate'])
        self.my_schedule = LinearSchedule(start_value=params['epsilon_start_value'],
                                 end_value=params['epsilon_end_value'],
                                 duration=params['epsilon_duration'])
        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])
        self.total_steps = 0 

    def get_action(self, obs, eps):
        if np.random.random() < eps:
            valid_actions = self.env.valid_actions()
            if not valid_actions:
                return 0
            action = np.random.choice(valid_actions)
            return action
        else:
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                q_values = self.behavior_policy_net(obs)
                
                # Filter for only valid actions
                valid_actions = self.env.valid_actions()
                if not valid_actions:
                    return 0  
                    
                # valid q values
                valid_q_values = [q_values[0, a].item() if a in valid_actions else -float('inf') 
                                  for a in range(self.action_dim)]
                
                action = np.argmax(valid_q_values)
            return action
    
    # Have x here to go along with experiment runner 
    def policy(self, obs, x=True):
        obs_tensor = self._arr_to_tensor(obs).view(1, -1)
        with torch.no_grad():
            q_values = self.behavior_policy_net(obs_tensor)
            
            # Filter for valid actions
            valid_actions = self.env.valid_actions()
            if not valid_actions:
                return 0 
                
            valid_q_values = [q_values[0, a].item() if a in valid_actions else -float('inf') 
                             for a in range(self.action_dim)]
            action = np.argmax(valid_q_values)
        return action

    def update_behavior_policy(self, batch_data):
        # Check if batch data is valid
        if len(batch_data[0]) == 0:
            return 0.0  
            
        batch_data_tensor = self._batch_to_tensor(batch_data)
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']
        
        # Forward pass to get q values 
        all_q_values = self.behavior_policy_net(obs_tensor)
        q_values = all_q_values.gather(1, actions_tensor)
        
        with torch.no_grad():
            # Get target q values
            next_q_values = self.target_policy_net(next_obs_tensor)
            next_q_values_max = next_q_values.max(dim=1)[0]
            # Compute target
            td_target = rewards_tensor + (1 - dones_tensor) * self.params['gamma'] * next_q_values_max.unsqueeze(1)

        # Compute TD loss
        td_loss = F.mse_loss(q_values, td_target)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        td_loss.backward()
        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.behavior_policy_net.parameters(), 10)
        self.optimizer.step()

        return td_loss.item()

    def update_target_policy(self):
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    def _arr_to_tensor(self, arr):
        arr = np.array(arr, dtype=np.float32)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor
    
    def train(self, num_episodes):
        train_returns = []
        train_loss = []     
        episode_count = 0
        
        # For reporting progress
        print_interval = max(1, num_episodes // 10)
        
        while episode_count < num_episodes:
            episode_rewards = []
            obs = self.env.reset()
            done = False
            episode_loss = []
            
            # Play one episode
            while not done:
                eps_t = self.my_schedule.get_value(self.total_steps)
                action = self.get_action(obs, eps_t)
                next_obs, reward, done = self.env.step(action)
                
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                episode_rewards.append(reward)
                obs = next_obs
                
                # Perform learning step
                if len(self.replay_buffer) >= self.params['batch_size'] and self.total_steps > 0:
                    if self.total_steps % self.params['freq_update_behavior_policy'] == 0:
                        batch_data = self.replay_buffer.sample_batch(self.params['batch_size'])
                        loss = self.update_behavior_policy(batch_data)
                        episode_loss.append(loss)
                    
                    if self.total_steps % self.params['freq_update_target_policy'] == 0:
                        self.update_target_policy()
                
                self.total_steps += 1
            
            # calculate return
            G = 0
            for r in reversed(episode_rewards):
                G = r + self.params['gamma'] * G
                
            train_returns.append(G)
            
            # Store average loss for this episode
            if episode_loss:
                train_loss.append(sum(episode_loss) / len(episode_loss))
            else:
                train_loss.append(0.0)
                
            # Progress reporting
            if episode_count % print_interval == 0 or episode_count == num_episodes - 1:
                avg_return = sum(train_returns[-print_interval:]) / min(print_interval, len(train_returns[-print_interval:]))
                avg_loss = sum(train_loss[-print_interval:]) / min(print_interval, len(train_loss[-print_interval:]))
                print(f"Episode {episode_count}/{num_episodes}: Avg Return = {avg_return:.4f}, Avg Loss = {avg_loss:.4f}, Epsilon = {eps_t:.4f}")
            
            episode_count += 1