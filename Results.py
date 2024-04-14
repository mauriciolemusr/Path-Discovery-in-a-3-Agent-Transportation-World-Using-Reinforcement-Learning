import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch


class ResultPrinter:
    def print_results(self, agents, total_rewards, total_distances, total_successes, num_steps):
        # Print experiment results for each agent
        print("Experiment Results:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards[i]:.2f}")
            print(f"  Total Success: {total_successes[i]}")
            print(f"  Total Distance: {total_distances[i]}")
            print(f"  Average Reward per Step: {total_rewards[i] / num_steps:.4f}")
            print(f"  Average Success per Step: {total_successes[i] / num_steps:.4f}")
            print(f"  Average Distance per Step: {total_distances[i] / num_steps:.2f}")
            print()

        # Print final Q-tables for each agent
        print("Final Q-Tables:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            q_table_dict = {}
            for state, actions in agent.q_table.items():
                agent_pos, pickup_pos, dropoff_pos = state
                state_str = f"Agent: {agent_pos}"
                q_table_dict[state_str] = actions

            q_table_df = pd.DataFrame.from_dict(q_table_dict, orient='index')
            q_table_df.index.name = 'State'
            q_table_df = q_table_df.apply(lambda x: x.apply(lambda y: '{:.3f}'.format(y)))
            print(q_table_df.to_string(index=True))
            print()

    def print_results_experiment2(self, agents, total_rewards, total_distances, total_successes, num_steps):
        # Print experiment 2 results
        print("Experiment 2 Results:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards[i]:.2f}")
            print(f"  Total Success: {total_successes[i]}")
            print(f"  Total Distance: {total_distances[i]}")
            print(f"  Average Reward per Step: {total_rewards[i] / num_steps:.4f}")
            print(f"  Average Success per Step: {total_successes[i] / num_steps:.4f}")
            print(f"  Average Distance per Step: {total_distances[i] / num_steps:.2f}")
            print()

        # Print final Q-tables for experiment 2
        print("Final Q-Tables for Experiment 2:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            q_table_dict = {}
            for state, actions in agent.q_table.items():
                agent_pos, pickup_pos, dropoff_pos = state
                state_str = f"Agent: {agent_pos}"
                q_table_dict[state_str] = actions

            q_table_df = pd.DataFrame.from_dict(q_table_dict, orient='index')
            q_table_df.index.name = 'State'
            q_table_df = q_table_df.apply(lambda x: x.apply(lambda y: '{:.3f}'.format(y)))
            print(q_table_df.to_string(index=True))
            print()

    def print_results_experiment3(self, agents, total_rewards, total_distances, total_successes, num_steps, learning_rate):
        # Print experiment 3 results with learning rate
        print(f"Experiment 3 Results (Learning Rate: {learning_rate}):")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards[i]:.2f}")
            print(f"  Total Success: {total_successes[i]}")
            print(f"  Total Distance: {total_distances[i]}")
            print(f"  Average Reward per Step: {total_rewards[i] / num_steps:.4f}")
            print(f"  Average Success per Step: {total_successes[i] / num_steps:.4f}")
            print(f"  Average Distance per Step: {total_distances[i] / num_steps:.2f}")
            print()

        # Print final Q-tables for experiment 3 with learning rate
        print(f"Final Q-Tables for Experiment 3 (Learning Rate: {learning_rate}):")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            q_table_dict = {}
            for state, actions in agent.q_table.items():
                agent_pos, pickup_pos, dropoff_pos = state
                state_str = f"Agent: {agent_pos}"
                q_table_dict[state_str] = actions

            q_table_df = pd.DataFrame.from_dict(q_table_dict, orient='index')
            q_table_df.index.name = 'State'
            q_table_df = q_table_df.apply(lambda x: x.apply(lambda y: '{:.3f}'.format(y)))
            print(q_table_df.to_string(index=True))
            print()

    def print_results_experiment4(self, agents, total_rewards_before, total_distances_before, total_successes_before, total_steps_before, total_rewards_after, total_distances_after, total_successes_after, total_steps_after):
        # Print experiment 4 results
        print("Experiment 4 Results:")
        print("Before changing pickup locations:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards_before[i]:.2f}")
            print(f"  Total Success: {total_successes_before[i]}")
            print(f"  Total Distance: {total_distances_before[i]}")
            print(f"  Average Reward per Step: {total_rewards_before[i] / total_steps_before:.4f}")
            print(f"  Average Success per Step: {total_successes_before[i] / total_steps_before:.4f}")
            print(f"  Average Distance per Step: {total_distances_before[i] / total_steps_before:.2f}")
            print()

        print("After changing pickup locations:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards_after[i]:.2f}")
            print(f"  Total Success: {total_successes_after[i]}")
            print(f"  Total Distance: {total_distances_after[i]}")
            print(f"  Average Reward per Step: {total_rewards_after[i] / total_steps_after:.4f}")
            print(f"  Average Success per Step: {total_successes_after[i] / total_steps_after:.4f}")
            print(f"  Average Distance per Step: {total_distances_after[i] / total_steps_after:.2f}")
            print()

        # Print final Q-tables for experiment 4
        print("Final Q-Tables for Experiment 4:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            q_table_dict = {}
            for state, actions in agent.q_table.items():
                agent_pos, pickup_pos, dropoff_pos = state
                state_str = f"Agent: {agent_pos}"
                q_table_dict[state_str] = actions

            q_table_df = pd.DataFrame.from_dict(q_table_dict, orient='index')
            q_table_df.index.name = 'State'
            q_table_df = q_table_df.apply(lambda x: x.apply(lambda y: '{:.3f}'.format(y)))
            print(q_table_df.to_string(index=True))
            print()


    def visualize_attractive_paths1(self, agents, environment):    
        num_agents = len(agents)
        fig, axs = plt.subplots(1, num_agents, figsize=(5*num_agents, 5))
        fig.suptitle('Agent Paths Visualization')

        for i, agent in enumerate(agents):
            ax = axs[i] if num_agents > 1 else axs
            ax.imshow(environment.grid, cmap='viridis', interpolation='nearest', vmin=-1, vmax=num_agents)

            # Plot agent's path
            for j in range(len(agent.path) - 1):
                current_pos = agent.path[j]
                next_pos = agent.path[j + 1]
                ax.quiver(current_pos[1], current_pos[0], next_pos[1] - current_pos[1], next_pos[0] - current_pos[0],
                        angles='xy', scale_units='xy', scale=1, color=f'C{i}', width=0.004, headwidth=3, headlength=4)

            # Plot starting point
            start_pos = agent.path[0]
            ax.plot(start_pos[1], start_pos[0], marker='s', markersize=10, color=f'C{i}', label=f'Agent {i+1} Start')

            # Plot ending point
            end_pos = agent.path[-1]
            ax.plot(end_pos[1], end_pos[0], marker='o', markersize=10, color=f'C{i}', label=f'Agent {i+1} End')

            ax.set_title(f'Agent {i+1} Path')
            ax.legend()

            # Set axis labels
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows')

        plt.tight_layout()
        plt.show()

    
    def visualize_attractive_paths2(self, agents, environment):
        fig, axs = plt.subplots(1, len(agents), figsize=(10, 4))  # Create subplots for each agent

        for idx, agent in enumerate(agents):
            grid = np.zeros(environment.grid_size)

            # Mark starting and ending points
            start_pos = agent.path[0]
            end_pos = agent.path[-1]
            if isinstance(end_pos, tuple):
                row, col = end_pos
                grid[start_pos] = agent.id + 2
                grid[row, col] = -1
            else:
                grid[start_pos] = agent.id + 2
                grid[end_pos] = -1

            # Draw arrows for the agent's path and color cells
            for i in range(len(agent.path) - 1):
                current_pos = agent.path[i]
                next_pos = agent.path[i + 1]
                if isinstance(next_pos, tuple):
                    dx = next_pos[1] - current_pos[1]
                    dy = next_pos[0] - current_pos[0]
                    arrow = FancyArrowPatch(
                        (current_pos[1], current_pos[0]),  # Start position
                        (next_pos[1], next_pos[0]),  # End position
                        color='lavender', arrowstyle='->', mutation_scale=10, linewidth=1  # Adjust arrow properties
                    )
                    axs[idx].add_patch(arrow)
                    row, col = current_pos
                    grid[row, col] = 1

            # Set colors for visualization
            cmap = plt.cm.get_cmap('viridis', len(agents) + 2)
            cmap.set_under('white')
            cmap.set_bad('red')

            # Visualize the grid
            axs[idx].imshow(grid, cmap=cmap, interpolation='nearest', vmin=-1, vmax=len(agents) + 1)

            # Add gridlines
            axs[idx].grid(color='black', linewidth=1)

            # Add legend below the subplot
            axs[idx].legend(handles=[plt.Line2D([0], [0], marker='s', color='w', label='Starting Point', markerfacecolor='black', markersize=15), 
                                     plt.Line2D([0], [0], marker='o', color='w', label='Ending Point', markerfacecolor='red', markersize=15)], 
                            loc='upper center', bbox_to_anchor=(0.5, -0.2))  # Adjust legend position

            axs[idx].set_title(f'Agent {idx+1}')  # Set subplot title

            # Mark starting and ending points
            axs[idx].scatter(start_pos[1], start_pos[0], color='black', marker='s', s=100, label='Starting Point')  # Mark starting point
            axs[idx].scatter(end_pos[1], end_pos[0], color='red', marker='o', s=100, label='Ending Point')  # Mark ending point
            
        plt.tight_layout()
        plt.show()


    def get_agent_paths(self, agents):
        """
        Get the paths of each agent and returns dictionary containing 
        the paths of each agent, with agent ID as keys and paths as values.
        """
        agent_paths = {}
        for agent in agents:
            agent_paths[agent.id] = agent.path
        return agent_paths