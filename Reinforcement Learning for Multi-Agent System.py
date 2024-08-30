import numpy as np
import random

# Define the environment
class GridWorld:
    def __init__(self, size, goals):
        self.size = size
        self.goals = goals
        self.state = None
        self.reset()

    def reset(self):
        self.state = (0, 0)  # Agents start at the top-left corner
        return self.state

    def step(self, agent_pos, action):
        next_state = list(agent_pos)
        if action == 0:   # Move up
            next_state[0] = max(0, agent_pos[0] - 1)
        elif action == 1: # Move down
            next_state[0] = min(self.size - 1, agent_pos[0] + 1)
        elif action == 2: # Move left
            next_state[1] = max(0, agent_pos[1] - 1)
        elif action == 3: # Move right
            next_state[1] = min(self.size - 1, agent_pos[1] + 1)

        reward = 0
        if tuple(next_state) == self.goals:
            reward = 10  # Reward for reaching the goal
        return tuple(next_state), reward

# Define the multi-agent system
class MultiAgentQLearning:
    def __init__(self, n_agents, grid_size, goals, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_agents = n_agents
        self.grid = GridWorld(grid_size, goals)
        self.q_tables = [np.zeros((grid_size, grid_size, 4)) for _ in range(n_agents)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, q_table):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(q_table[state[0], state[1], :])  # Exploit

    def update_q_table(self, agent_idx, state, action, reward, next_state):
        q_predict = self.q_tables[agent_idx][state[0], state[1], action]
        q_target = reward + self.gamma * np.max(self.q_tables[agent_idx][next_state[0], next_state[1], :])
        self.q_tables[agent_idx][state[0], state[1], action] += self.alpha * (q_target - q_predict)

    def train(self, episodes):
        for episode in range(episodes):
            states = [self.grid.reset() for _ in range(self.n_agents)]
            done = False
            while not done:
                for agent_idx in range(self.n_agents):
                    action = self.choose_action(states[agent_idx], self.q_tables[agent_idx])
                    next_state, reward = self.grid.step(states[agent_idx], action)
                    self.update_q_table(agent_idx, states[agent_idx], action, reward, next_state)
                    states[agent_idx] = next_state
                    if states[agent_idx] == self.grid.goals:
                        done = True

    def get_q_tables(self):
        return self.q_tables

# Simulation parameters
n_agents = 2
grid_size = 5
goals = (4, 4)
episodes = 1000

# Create the multi-agent system
multi_agent_system = MultiAgentQLearning(n_agents, grid_size, goals)

# Train the agents
multi_agent_system.train(episodes)

# Print Q-tables
q_tables = multi_agent_system.get_q_tables()
for i, q_table in enumerate(q_tables):
    print(f"Q-Table for Agent {i + 1}:")
    print(q_table)
