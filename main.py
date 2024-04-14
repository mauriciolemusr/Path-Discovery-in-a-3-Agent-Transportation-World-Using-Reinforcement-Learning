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

    environment.visualize()  

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

    # Run Experiment 1a
    print("Experiment 1a:")
    total_rewards_1a, total_distances_1a, total_successes_1a = experiments.run_experiment1a(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1a, total_distances_1a, total_successes_1a, num_steps)
    # Visualize attractive paths and position frquencies for Experiment 1a
  #  result_printer.visualize_attractive_paths(agents, environment)
    #result_printer.visualize_position_freq(environment)

    # Run Experiment 1b
    print("Experiment 1b:")
    total_rewards_1b, total_distances_1b, total_successes_1b = experiments.run_experiment1b(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1b, total_distances_1b, total_successes_1b, num_steps)
    # Visualize attractive paths and position frquencies for Experiment 1b
    #result_printer.visualize_attractive_paths(agents, environment)
   # result_printer.visualize_position_freq(environment)

    # Run Experiment 1c
    print("Experiment 1c:")
    total_rewards_1c, total_distances_1c, total_successes_1c = experiments.run_experiment1c(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results(agents, total_rewards_1c, total_distances_1c, total_successes_1c, num_steps)
    # Visualize attractive paths and position frquencies for Experiment 1c
   # result_printer.visualize_attractive_paths(agents, environment)
   # result_printer.visualize_position_freq(environment)

    # Run Experiment 2
    print("Experiment 2:")
    total_rewards_2, total_distances_2, total_successes_2 = experiments.run_experiment2(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results_experiment2(agents, total_rewards_2, total_distances_2, total_successes_2, num_steps)
    # Visualize attractive paths and position frquencies for Experiment 2
    #result_printer.visualize_attractive_paths(agents, environment)
   # result_printer.visualize_position_freq(environment)

    # Run Experiment 3 with different learning rates
    learning_rates = [0.15, 0.45]
    for learning_rate in learning_rates:
        print(f"Experiment 3 (Learning Rate: {learning_rate}):")
        total_rewards_3, total_distances_3, total_successes_3 = experiments.run_experiment3(environment, agents, num_steps, initial_agent_positions, learning_rate)
        result_printer.print_results_experiment3(agents, total_rewards_3, total_distances_3, total_successes_3, num_steps, learning_rate)
        # Visualize attractive paths and position frquencies for Experiment 3
       # result_printer.visualize_attractive_paths(agents, environment)
        #result_printer.visualize_position_freq(environment)

    # Run Experiment 4
    print("Experiment 4:")
    (
        total_rewards_before_4, total_distances_before_4, total_successes_before_4, total_steps_before_4,
        total_rewards_after_4, total_distances_after_4, total_successes_after_4, total_steps_after_4,
    ) = experiments.run_experiment4(environment, agents, num_steps, initial_agent_positions)
    result_printer.print_results_experiment4(
        agents, total_rewards_before_4, total_distances_before_4, total_successes_before_4, total_steps_before_4,
        total_rewards_after_4, total_distances_after_4, total_successes_after_4, total_steps_after_4,
    )

    # Visualize attractive paths and position frquencies for Experiment 4
    #result_printer.visualize_attractive_paths(agents, environment)
    #result_printer.visualize_position_freq(environment)

if __name__ == "__main__":
    main() 