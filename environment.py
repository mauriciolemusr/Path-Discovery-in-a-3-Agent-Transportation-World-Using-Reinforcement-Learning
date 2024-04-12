import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, grid_size, pickup_locations, dropoff_locations):
        self.grid_size = grid_size
        self.pickup_locations = pickup_locations
        self.dropoff_locations = dropoff_locations
        self.grid = np.zeros(grid_size, dtype=int)
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def reset(self, initial_agent_positions):
        self.grid = np.zeros(self.grid_size, dtype=int)
        for pickup_location in self.pickup_locations:
            self.grid[pickup_location] = 1
        for i, agent in enumerate(self.agents):
            agent.position = initial_agent_positions[i]
            agent.reset()

    def step(self):
        for agent in self.agents:
            action = agent.choose_action(self)
            reward = self.execute_action(agent, action)
            agent.update_q_table(self, action, reward)

    def execute_action(self, agent, action):
        x, y = agent.position

        if action == 'up':
            new_x, new_y = x, y - 1
        elif action == 'down':
            new_x, new_y = x, y + 1
        elif action == 'left':
            new_x, new_y = x - 1, y
        elif action == 'right':
            new_x, new_y = x + 1, y
        else:
            new_x, new_y = x, y

        if self.is_valid_position(new_x, new_y):
            agent.position = (new_x, new_y)

        if action == 'pickup' and self.grid[x, y] == 1 and not agent.carrying_block:
            self.grid[x, y] = 0
            agent.carrying_block = True
        elif action == 'dropoff' and (x, y) in self.dropoff_locations and agent.carrying_block:
            agent.carrying_block = False

        reward = self.calculate_reward(agent)
        return reward

    def is_valid_position(self, x, y):
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        for agent in self.agents:
            if agent.position == (x, y):
                return False
        return True

    def calculate_reward(self, agent):
        x, y = agent.position

        if agent.carrying_block and (x, y) in self.dropoff_locations:
            return 10.0   # Reward for successful dropoff
        elif not agent.carrying_block and (x, y) in self.pickup_locations and self.grid[x, y] == 1:
            return 1.0  # Reward for successful pickup
        else:
            return 0.0  # No reward or penalty for other steps

    def get_state(self, agent):
        agent_position = agent.position
        pickup_locations = tuple(sorted(self.pickup_locations))
        dropoff_locations = tuple(sorted(self.dropoff_locations))
        return (agent_position, pickup_locations, dropoff_locations)

    def get_available_actions(self, agent):
        actions = ['up', 'down', 'left', 'right']
        x, y = agent.position

        if self.grid[x, y] == 1 and not agent.carrying_block:
            actions.append('pickup')
        if agent.carrying_block and (x, y) in self.dropoff_locations:
            actions.append('dropoff')

        return actions

    def is_terminal_state(self):
        if not any(self.grid[location] == 1 for location in self.pickup_locations) and \
                all(agent.carrying_block == False for agent in self.agents):
            return True
        return False

    def visualize(self):
        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap='binary')

        for pickup_location in self.pickup_locations:
            ax.text(pickup_location[1], pickup_location[0], 'P', ha='center', va='center', color='blue')

        for dropoff_location in self.dropoff_locations:
            ax.text(dropoff_location[1], dropoff_location[0], 'D', ha='center', va='center', color='green')

        for agent in self.agents:
            x, y = agent.position
            ax.text(y, x, 'A', ha='center', va='center', color='red')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Environment')

        plt.show() 