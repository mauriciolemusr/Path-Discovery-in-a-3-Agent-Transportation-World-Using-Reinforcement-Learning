import numpy as np 
import pandas as pd
import random 
from Environment import Environment
from Agent import Agent
from Experiments import Experiments
from Results import ResultPrinter
  
def main():
    # Defining the agents and the environment
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

    # environment.visualize() #NEED TO FIX THIS

    num_steps = 9000
    experiments = Experiments()

    seed_input = input("Enter a seed value (or press Enter to generate a random seed): ")
    if seed_input == "":
        seed = random.randint(1, 10000)
        print(f"Generated Random Seed: {seed}")
    else:
        seed = int(seed_input)
        print(f"Using Provided Seed: {seed}")

    np.random.seed(seed)
    random.seed(seed)

    result_printer = ResultPrinter()

    print("Experiment 1a:")
    total_rewards_1a, total_distances_1a, total_successes_1a = experiments.run_experiment1a(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1a, total_distances_1a, total_successes_1a, num_steps)
    
    result_printer.visualize_attractive_paths(agents, environment)

    print("Experiment 1b:")
    total_rewards_1b, total_distances_1b, total_successes_1b = experiments.run_experiment1b(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1b, total_distances_1b, total_successes_1b, num_steps)

    print("Experiment 1c:")
    total_rewards_1c, total_distances_1c, total_successes_1c = experiments.run_experiment1c(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1c, total_distances_1c, total_successes_1c, num_steps)

    print("Experiment 2:")
    total_rewards_2, total_distances_2, total_successes_2 = experiments.run_experiment2(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results_experiment2(agents, total_rewards_2, total_distances_2, total_successes_2, num_steps)

    learning_rates = [0.15, 0.45]
    for learning_rate in learning_rates:
        print(f"Experiment 3 (Learning Rate: {learning_rate}):")
        total_rewards_3, total_distances_3, total_successes_3 = experiments.run_experiment3(environment, agents, num_steps, initial_agent_positions, learning_rate)
        result_printer.print_results_experiment3(agents, total_rewards_3, total_distances_3, total_successes_3, num_steps, learning_rate)

    print("Experiment 4:")
    (
        total_rewards_before_4, total_distances_before_4, total_successes_before_4, total_steps_before_4,
        total_rewards_after_4, total_distances_after_4, total_successes_after_4, total_steps_after_4,
    ) = experiments.run_experiment4(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results_experiment4(
        agents, total_rewards_before_4, total_distances_before_4, total_successes_before_4, total_steps_before_4,
        total_rewards_after_4, total_distances_after_4, total_successes_after_4, total_steps_after_4,
    )


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


Regarding prefered path for agents, we can look at Q-Table to see which actions have the highest 
Q-Values for each state.

For Experiment 3 we chose to rerun Experiment 1c with different learning rates.

""" 
