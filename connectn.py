import numpy as np

class ConnectN():
    def __init__(self, n: int) -> None:
        self.n = n
        all_dims = {
            3: (5, 6),
            4: (6, 7),
            5: (7, 8),
            6: (8, 9)
        }
        # if invalid n play connect 4
        self.dim = all_dims.get(self.n, (6, 7))
        self.board = np.zeros(self.dim, dtype=int)
        
        # 1 for agent, -1 for opponent
        self.current_player = 1  
        self.last_action = None
        self.done = False
    
    def reset(self):
        self.board = np.zeros(self.dim, dtype=int)
        self.current_player = 1 
        self.last_action = None
        self.done = False
        return self.board.copy().flatten()
    
    def valid_actions(self):
        return [col for col in range(self.dim[1]) if self.board[0, col] == 0]
    
    def random_action(self):
        valid_actions = self.valid_actions()
        if valid_actions:
            return np.random.choice(valid_actions)
        return None

    def step(self, action: int):
        if self.done:
            return self.board.copy().flatten(), 0, True
            
        if action not in self.valid_actions():
            return self.board.copy().flatten(), -1, False
            
        # Find the lowest empty row in the selected column
        for row in range(self.dim[0]-1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                self.last_action = (row, action)
                break
                
        # Check if the current player has won
        if self._check_win(self.current_player):
            self.done = True
            reward = self.current_player
            return self.board.copy().flatten(), reward, True
            
        # Check if the board is full (draw)
        if len(self.valid_actions()) == 0:
            self.done = True
            return self.board.copy().flatten(), 0, True
            
        # Switch player
        self.current_player *= -1
        
        return self.board.copy().flatten(), 0, False
    
    def _check_win(self, player):
        if self.last_action is None:
            return False
            
        row, col = self.last_action
        
        # Check horizontally
        for c in range(max(0, col - self.n + 1), min(col + 1, self.dim[1] - self.n + 1)):
            if np.all(self.board[row, c:c+self.n] == player):
                return True
                
        # Check vertically
        for r in range(max(0, row - self.n + 1), min(row + 1, self.dim[0] - self.n + 1)):
            if np.all(self.board[r:r+self.n, col] == player):
                return True
                
        # Check diagonal (top-left to bottom-right)
        for i in range(-self.n + 1, self.n):
            diag = np.array([self.board[row+j, col+j] for j in range(-i, -i+self.n) 
                             if 0 <= row+j < self.dim[0] and 0 <= col+j < self.dim[1]])
            if len(diag) == self.n and np.all(diag == player):
                return True
                
        # Check diagonal (top-right to bottom-left)
        for i in range(-self.n + 1, self.n):
            diag = np.array([self.board[row+j, col-j] for j in range(-i, -i+self.n) 
                             if 0 <= row+j < self.dim[0] and 0 <= col-j < self.dim[1]])
            if len(diag) == self.n and np.all(diag == player):
                return True
                
        return False
    
    def simulate_episode(self, agent_policy_fn):           
        states = []
        actions = []
        rewards = []
        
        state = self.reset()
        done = False
        
        while not done:
            if self.current_player == 1:
                action = agent_policy_fn(state)
                states.append(state.copy())
                actions.append(action)
            else:
                # Opponent only takes random actions
                action = self.random_action()
                
            next_state, reward, done = self.step(action)
            
            rewards.append(reward)  
            state = next_state
            
        return states, actions, rewards