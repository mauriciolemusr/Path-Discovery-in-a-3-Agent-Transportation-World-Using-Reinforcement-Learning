import pandas as pd

class ResultPrinter:
    def print_results(self, agents, total_rewards, total_distances, total_successes, num_steps):
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