import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            q_table_df.sort_index(inplace=True)
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
            q_table_df.sort_index(inplace=True)
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
            q_table_df.sort_index(inplace=True)
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
            if total_steps_before != 0:
                print(f"  Average Reward per Step: {total_rewards_before[i] / total_steps_before:.4f}")
                print(f"  Average Success per Step: {total_successes_before[i] / total_steps_before:.4f}")
                print(f"  Average Distance per Step: {total_distances_before[i] / total_steps_before:.2f}")
            else:
                print("  Average Reward per Step: N/A")
                print("  Average Success per Step: N/A")
                print("  Average Distance per Step: N/A")
            print()

        print("After changing pickup locations:")
        for i, agent in enumerate(agents):
            print(f"Agent {i + 1}:")
            print(f"  Total Reward: {total_rewards_after[i]:.2f}")
            print(f"  Total Success: {total_successes_after[i]}")
            print(f"  Total Distance: {total_distances_after[i]}")
            if total_steps_after != 0:
                print(f"  Average Reward per Step: {total_rewards_after[i] / total_steps_after:.4f}")
                print(f"  Average Success per Step: {total_successes_after[i] / total_steps_after:.4f}")
                print(f"  Average Distance per Step: {total_distances_after[i] / total_steps_after:.2f}")
            else:
                print("  Average Reward per Step: N/A")
                print("  Average Success per Step: N/A")
                print("  Average Distance per Step: N/A")
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
            q_table_df.sort_index(inplace=True)
            print(q_table_df.to_string(index=True))
            print()

    # visualize_position_freq function visualizes the frequency each agent travels across the entire experiment as separate heatmaps
    def visualize_position_freq(self, environment):
        for i, agent in enumerate(environment.agents):            
            plt.figure()
            if i == 0:
                plt.title("Position Frequency for Agent \"Black\"")
                print("Position Frequency Map for Agent \"Black\":")
            elif i == 1:
                plt.title("Position Frequency for Agent \"Red\"")
                print("Position Frequency Map for Agent \"Red\":")
            elif i == 2:
                plt.title("Position Frequency for Agent \"Blue\"")
                print("Position Frequency Map for Agent \"Blue\":")
            sns.heatmap(agent.position_frequency, annot=True, fmt='g')
            plt.show()        

    # visualize_attractive_paths function visualizes the most attractive paths for each grid space based on highest q-table value - NOT USED
    def visualize_attractive_paths(self, agents, environment):
        """
        Visualize attractive paths based on learned Q-values.
        """
        for i, agent in enumerate(agents):
            print(f"Attractive Paths for Agent {i + 1}:")
            q_values = agent.q_table

            # Identify most promising actions for each state
            attractive_paths = {}
            for state, actions in q_values.items():
                agent_pos, _, _ = state
                if agent_pos not in attractive_paths:
                    attractive_paths[agent_pos] = []
                max_q_value = max(actions.values())
                best_actions = [action for action, q_value in actions.items() if q_value == max_q_value]
                attractive_paths[agent_pos].extend(best_actions)

            # Plot environment grid
            plt.figure(figsize=(5, 5))
            plt.imshow(environment.grid, cmap='binary')

            # Highlight attractive paths
            for position, actions in attractive_paths.items():
                x, y = position
                for action in actions:
                    if action == 'up':
                        plt.arrow(y, x, 0, -0.4, head_width=0.1, head_length=0.1, fc='red', ec='red')
                    elif action == 'down':
                        plt.arrow(y, x, 0, 0.4, head_width=0.1, head_length=0.1, fc='red', ec='red')
                    elif action == 'left':
                        plt.arrow(y, x, -0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
                    elif action == 'right':
                        plt.arrow(y, x, 0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')

            plt.title(f"Attractive Paths for Agent {i + 1}")
            plt.axis('off')
            plt.show() 