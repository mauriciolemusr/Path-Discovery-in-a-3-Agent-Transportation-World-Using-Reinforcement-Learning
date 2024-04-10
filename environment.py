import numpy as np

class Environment:
    def __init__(self, grid_size, pickup_locations, dropoff_locations):
        """
        Initializes the environment.
        grid_size: tuple (width, height) representing the size of the grid
        pickup_locations: list of tuples (x, y) representing the pickup locations
        dropoff_locations: list of tuples (x, y) representing the dropoff locations
        """
        self.grid_size = grid_size
        self.pickup_locations = pickup_locations
        self.dropoff_locations = dropoff_locations
        self.grid = np.zeros(grid_size, dtype=int)
        self.agents = []

    def add_agent(self, agent):
        """
        Adds an agent object to the environment.
        """
        self.agents.append(agent)



    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.grid = np.zeros(self.grid_size, dtype=int)
        for pickup_location in self.pickup_locations:
            self.grid[pickup_location] = 1  # 1 represents a block
        for agent in self.agents:
            agent.reset()



    def step(self):
        """
        Performs one step in the environment, allowing each agent to take an action.
        """
        for agent in self.agents:
            action = agent.choose_action(self)
            reward = self.execute_action(agent, action)
            agent.update_q_table(self, action, reward)



    def execute_action(self, agent, action):
        """
        Executes the given action for the agent and updates the environment.
        str representing the action to take
        ('up', 'down', 'left', 'right', 'pickup', 'dropoff')
        """
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
        """
        Checks if the given position is valid (within the grid and not occupied by another agent).
        int representing the x-coordinate
        int representing the y-coordinate
        returns bool indicating if the position is valid
        """
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        for agent in self.agents:
            if agent.position == (x, y):
                return False
        return True



    def calculate_reward(self, agent):
        """
        Calculates the reward for the agent based on its current position and action.
        returns float representing the reward
        """
        x, y = agent.position

        if agent.carrying_block and (x, y) in self.dropoff_locations:
            return 10.0  # Reward for successfully dropping off a block
        elif not agent.carrying_block and (x, y) in self.pickup_locations and self.grid[x, y] == 1:
            return 1.0  # Reward for successfully picking up a block
        else:
            return -0.1  # Small negative reward for each step to encourage efficiency



    def get_state(self, agent):
        """
        Gets the current state of the environment from the perspective of the given agent.
        returns tuple representing the state (agent_position, block_positions, dropoff_positions)
        """
        agent_position = agent.position
        block_positions = self.pickup_locations
        dropoff_positions = self.dropoff_locations
        return (agent_position, block_positions, dropoff_positions)



    def get_available_actions(self, agent):
        """
        Gets the available actions for the given agent in the current state.
        returns list of str representing the available actions
        """
        actions = ['up', 'down', 'left', 'right']
        x, y = agent.position

        if self.grid[x, y] == 1 and not agent.carrying_block:
            actions.append('pickup')
        if agent.carrying_block and (x, y) in self.dropoff_locations:
            actions.append('dropoff')

        return actions
    


    def is_terminal_state(self):
        """
        Checks if the current state is a terminal state.
        :return: bool indicating if the current state is a terminal state
        """
        # Check if all blocks have been picked up and dropped off
        if not any(self.grid[location] == 1 for location in self.pickup_locations) and \
                all(agent.carrying_block == False for agent in self.agents):
            return True
        return False



    def visualize(self):
        """
        Visualizes the current state of the environment.
        """
        grid = [[' ' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]

        # Mark the pickup locations with 'P'
        for location in self.pickup_locations:
            x, y = location
            grid[x][y] = 'P'

        # Mark the dropoff locations with 'D'
        for location in self.dropoff_locations:
            x, y = location
            grid[x][y] = 'D'

        # Mark the agents with '!!'
        for agent in self.agents:
            x, y = agent.position
            grid[x][y] = '!!'

        # Print the grid with rows and columns
        print('   ' + '|'.join(f'{i:^3}' for i in range(self.grid_size[1])))
        print('  +' + '-' * (4 * self.grid_size[1] - 1) + '+')
        for i, row in enumerate(grid):
            print(f'{i:^2}|', end='')
            print('|'.join(f'{cell:^3}' for cell in row), end='|\n')
            if i < self.grid_size[0] - 1:
                print('  |' + '_' * (4 * self.grid_size[1] - 1) + '|')
        print('  +' + '-' * (4 * self.grid_size[1] - 1) + '+')
        