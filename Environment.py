import numpy as np 
import matplotlib.pyplot as plt  # For visualization
from matplotlib.colors import ListedColormap

class Environment:
    def __init__(self, grid_size, pickup_locations, dropoff_locations):
        # Initializing the Environment class with grid size, pickup and dropoff locations, an empty grid, and an empty list of agents.
        self.grid_size = grid_size
        self.pickup_locations = pickup_locations
        self.dropoff_locations = dropoff_locations
        self.grid = np.zeros(grid_size, dtype=int)
        self.agents = []
        self.dropoff_counts = {location: 0 for location in dropoff_locations}  

    # Method that adds an agent to the environment.
    def add_agent(self, agent):
        self.agents.append(agent)

    # Method for resetting the environment between experiments
    def reset_environment(self, initial_agent_positions):
        self.grid = np.zeros(self.grid_size, dtype=int)
        for pickup_location in self.pickup_locations:
            self.grid[pickup_location] = 5  
        for i, agent in enumerate(self.agents):
            agent.position = initial_agent_positions[i]
            agent.carrying_block = False  # Reset the carrying_block attribute
            agent.position_frequency = [[0 for j in range(0, 5)] for i in range(0, 5)]
        self.dropoff_counts = {location: 0 for location in self.dropoff_locations}  

    # Method for resetting the environment if terminal state reached by placing pickup locations, agents, and resetting agents' states.
    def reset(self, initial_agent_positions):
        self.grid = np.zeros(self.grid_size, dtype=int)
        for pickup_location in self.pickup_locations:
            self.grid[pickup_location] = 5  
        for i, agent in enumerate(self.agents):
            agent.position = initial_agent_positions[i]
            agent.carrying_block = False  # Reset the carrying_block attribute
        self.dropoff_counts = {location: 0 for location in self.dropoff_locations}  

    # Method for executing one step in the environment for each agent.
    def step(self):
        for agent in self.agents:
            action = agent.choose_action(self)
            reward = self.execute_action(agent, action)
            agent.update_q_table(self, action, reward)

    # Method for executing an action for the given agent and updating the environment accordingly.
    def execute_action(self, agent, action):
        x, y = agent.position

        # Determining the new position based on the action.
        if action == 'up':
            new_x, new_y = x - 1, y
        elif action == 'down':
            new_x, new_y = x + 1, y
        elif action == 'left':
            new_x, new_y = x, y - 1
        elif action == 'right':
            new_x, new_y = x, y + 1
        else:
            new_x, new_y = x, y

        # Updating the agent's position if the new position is valid.
        if self.is_valid_position(new_x, new_y):
            agent.position = (new_x, new_y)

        # Executing pickup and dropoff actions and calculating reward.
        if action == 'pickup' and self.grid[x, y] > 0 and not agent.carrying_block:  
            self.grid[x, y] -= 1  # Change this line
            agent.carrying_block = True
        elif action == 'dropoff' and (x, y) in self.dropoff_locations and agent.carrying_block and self.dropoff_counts[(x, y)] < 5:  # Change this line
            agent.carrying_block = False
            self.dropoff_counts[(x, y)] += 1  
        
        # Update frequency matrix after agent action
        agent.update_position_freq()

        reward = self.calculate_reward(agent)
        return reward

    # Checks if the position (x, y) is valid within the grid boundaries and not occupied by another agent.
    def is_valid_position(self, x, y):
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        for agent in self.agents:
            if agent.position == (x, y):
                return False
        return True

    # Calculates the reward for the agent's current position and state.
    def calculate_reward(self, agent):
        x, y = agent.position

        if agent.carrying_block and (x, y) in self.dropoff_locations and self.dropoff_counts[(x, y)] < 5:  # Change this line
            return 10.0   # Reward for successful dropoff
        elif not agent.carrying_block and (x, y) in self.pickup_locations and self.grid[x, y] > 0:
            return 1.0  # Reward for successful pickup
        else:
            return 0.0  # No reward or penalty for other steps

    # Gets the state representation for a given agent.
    def get_state(self, agent):
        agent_position = agent.position
        pickup_locations = tuple(sorted(self.pickup_locations))
        dropoff_locations = tuple(sorted(self.dropoff_locations))
        return (agent_position, pickup_locations, dropoff_locations)

    # Gets available actions for the given agent based on its current position and state.
    def get_available_actions(self, agent):
        actions = ['up', 'down', 'left', 'right']
        x, y = agent.position

        if self.grid[x, y] > 0 and not agent.carrying_block:
            actions.append('pickup')
        if agent.carrying_block and (x, y) in self.dropoff_locations and self.dropoff_counts[(x, y)] < 5:  # Change this line
            actions.append('dropoff')

        return actions

    # Checks if the environment has reached a terminal state where all pickups are delivered.
    def is_terminal_state(self):
        if all(count == 5 for count in self.dropoff_counts.values()):  
            return True
        return False


    # Visualizes the environment grid, pickup locations, drop-off locations, and agent positions from the environment class.
    def visualize(self):
        # Define the color map for the grid
        cmap = ListedColormap(['white'])
        
        # Create a figure and axis with a specified size
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the grid with custom color maps
        ax.matshow(self.grid, cmap=cmap)
        
        # Loop over the data dimensions and create text annotations
        for pickup_location in self.pickup_locations:
            ax.text(pickup_location[1], pickup_location[0], 'Pickup', ha='center', va='center', color='blue', fontsize=12, weight='bold')

        for dropoff_location in self.dropoff_locations:
            ax.text(dropoff_location[1], dropoff_location[0], 'Dropoff', ha='center', va='center', color='green', fontsize=12, weight='bold')

        for agent in self.agents:
            x, y = agent.position
            ax.text(y, x, 'Agent', ha='center', va='center', color='red', fontsize=12, weight='bold')
        
        # Draw grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.grid(which='major', color='white', linestyle='', linewidth=0)
        
        # Hide the major tick labels
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Set the title
        ax.set_title('Environment', fontsize=16, weight='bold')

        # Display the plot
        plt.show()

