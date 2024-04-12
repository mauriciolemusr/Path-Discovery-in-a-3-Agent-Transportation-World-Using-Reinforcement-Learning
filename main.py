import numpy as np
import pandas as pd
from environment import Environment
from agent import Agent
from experiments import Experiments

def print_results(agents, total_rewards, total_distances, total_successes, num_steps):
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

def main():
    grid_size = (5, 5)
    pickup_locations = [(0, 4), (1, 3), (4, 1)]
    dropoff_locations = [(0, 0), (2, 0), (3, 4)]
    environment = Environment(grid_size, pickup_locations, dropoff_locations)

    agents = [
        Agent((0, 2), learning_rate=0.3, discount_factor=0.5),
        Agent((2, 2), learning_rate=0.3, discount_factor=0.5),
        Agent((4, 2), learning_rate=0.3, discount_factor=0.5)
    ]

    initial_agent_positions = [(0, 2), (2, 2), (4, 2)]

    for agent in agents:
        environment.add_agent(agent)

    environment.visualize()

    num_steps = 9000
    experiments = Experiments()

    print("Experiment 1a:")
    total_rewards_1a, total_distances_1a, total_successes_1a = experiments.run_experiment1a(environment, agents, num_steps, initial_agent_positions)
    print_results(agents, total_rewards_1a, total_distances_1a, total_successes_1a, num_steps)

    print("Experiment 1b:")
    total_rewards_1b, total_distances_1b, total_successes_1b = experiments.run_experiment1b(environment, agents, num_steps, initial_agent_positions)
    print_results(agents, total_rewards_1b, total_distances_1b, total_successes_1b, num_steps)

    print("Experiment 1c:")
    total_rewards_1c, total_distances_1c, total_successes_1c = experiments.run_experiment1c(environment, agents, num_steps, initial_agent_positions)
    print_results(agents, total_rewards_1c, total_distances_1c, total_successes_1c, num_steps)

    # Add code for Experiments 2, 3, and 4

if __name__ == "__main__":
    main()

"""
NOTES FOR REPORT(REMOVE BEFORE SUBMISSION!!!):

Total Reward:
This is the sum of all the rewards the agent received throughout the entire experiment.
If the reward for a successful pickup is X and for a successful dropoff is Y, then Total Reward = (X * number of successful pickups) + (Y * number of successful dropoffs) + (negative reward for other steps)

Total Success:
This is the total number of times the agent was successful in either picking up a block or dropping it off at the correct location throughout the experiment.
Total Success = number of successful pickups + number of successful dropoffs

Total Distance:
This is the sum of the distances the agent moved during the entire experiment.
For each step, the distance is calculated as the Manhattan distance between the agent's starting position and ending position.
Total Distance = sum of Manhattan distances for all steps

Average Reward per Step:
This is the average reward the agent received per step during the experiment.
Average Reward per Step = Total Reward / Total Number of Steps

Average Success per Step:
This is the average number of times the agent was successful (picked up or dropped off a block) per step during the experiment.
Average Success per Step = Total Success / Total Number of Steps

Average Distance per Step:
This is the average distance the agent moved per step during the experiment.
Average Distance per Step = Total Distance / Total Number of Steps

""" 