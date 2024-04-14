import numpy as np 
import matplotlib.pyplot as plt  # For visualization
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

class Environment:
    def __init__(self, grid_size, pickup_locations, dropoff_locations):
        # Initializing the Environment class with grid size, pickup and dropoff locations, an empty grid, and an empty list of agents.
        self.grid_size = grid_size
        self.pickup_locations = pickup_locations
        self.dropoff_locations = dropoff_locations
        self.grid = np.zeros(grid_size, dtype=int)
        self.agents = []

    # Method that adds an agent to the environment.
    def add_agent(self, agent):
        self.agents.append(agent)

    # Method for resetting the environment by placing pickup locations, agents, and resetting agents' states.
    def reset(self, initial_agent_positions):
        self.grid = np.zeros(self.grid_size, dtype=int)
        for pickup_location in self.pickup_locations:
            self.grid[pickup_location] = 1
        for i, agent in enumerate(self.agents):
            agent.position = initial_agent_positions[i]
            agent.reset()

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
            new_x, new_y = x, y - 1
        elif action == 'down':
            new_x, new_y = x, y + 1
        elif action == 'left':
            new_x, new_y = x - 1, y
        elif action == 'right':
            new_x, new_y = x + 1, y
        else:
            new_x, new_y = x, y

        # Updating the agent's position if the new position is valid.
        if self.is_valid_position(new_x, new_y):
            agent.position = (new_x, new_y)

        # Executing pickup and dropoff actions and calculating reward.
        if action == 'pickup' and self.grid[x, y] == 1 and not agent.carrying_block:
            self.grid[x, y] = 0
            agent.carrying_block = True
        elif action == 'dropoff' and (x, y) in self.dropoff_locations and agent.carrying_block:
            agent.carrying_block = False

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

        if agent.carrying_block and (x, y) in self.dropoff_locations:
            return 10.0   # Reward for successful dropoff
        elif not agent.carrying_block and (x, y) in self.pickup_locations and self.grid[x, y] == 1:
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

        if self.grid[x, y] == 1 and not agent.carrying_block:
            actions.append('pickup')
        if agent.carrying_block and (x, y) in self.dropoff_locations:
            actions.append('dropoff')

        return actions

    # Checks if the environment has reached a terminal state where all pickups are delivered.
    def is_terminal_state(self):
        if not any(self.grid[location] == 1 for location in self.pickup_locations) and \
                all(agent.carrying_block == False for agent in self.agents):
            return True
        return False


    # Method to visualize the agent's path on the grid.
    def visualize_agent_path(self, agent, path):
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(-0.5, self.grid_size[0], 1))
        ax.set_yticks(np.arange(-0.5, self.grid_size[1], 1))
        ax.grid(which='both')
        ax.set_aspect('equal')
        ax.imshow(self.grid, cmap='Greys', origin='lower')

        # Mark starting point
        start_x, start_y = path[0]
        ax.plot(start_x, start_y, 'ro', markersize=10)

        # Mark ending point
        end_x, end_y = path[-1]
        ax.plot(end_x, end_y, 'bo', markersize=10)

        # Draw arrows and color cells
        for i in range(len(path) - 1):
            x, y = path[i]
            next_x, next_y = path[i + 1]
            dx = next_x - x
            dy = next_y - y
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

        # Set colors for visualization
        cmap = plt.cm.get_cmap('cool')
        norm = plt.Normalize(0, len(path))
        for i, (x, y) in enumerate(path):
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=cmap(norm(i))))

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Starting Point', markerfacecolor='r', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Ending Point', markerfacecolor='b', markersize=10),
            FancyArrowPatch((0,0), (1,1), color='black', label='Agent Path', arrowstyle='-|>', mutation_scale=15)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.show()


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