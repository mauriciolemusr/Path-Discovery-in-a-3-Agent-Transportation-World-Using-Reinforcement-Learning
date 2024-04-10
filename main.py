import numpy as np
from environment import Environment
from agent import Agent


def calculate_distance(position1, position2):
    """
    Calculates the Manhattan distance between two positions.
    :param position1: tuple (x, y) representing the first position
    :param position2: tuple (x, y) representing the second position
    :return: int representing the Manhattan distance
    """
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])

def print_q_tables(agents):
    """
    Prints the final Q-tables for each agent.
    :param agents: list of Agent objects
    """
    for i, agent in enumerate(agents):
        print(f"Q-Table: Agent {i + 1}")
        state_counter = 1
        for state, actions in agent.q_table.items():
            print(f"State {state_counter}: {state}")
            for action, q_value in actions.items():
                print(f"  Action: {action}, Q-value: {q_value:.3f}")
            state_counter += 1
            print()
        print()

def run_experiment(environment, agents, num_steps, policy):
    """
    Runs an experiment with the given environment, agents, number of steps, and policy.
    :param environment: Environment object
    :param agents: list of Agent objects
    :param num_steps: int representing the number of steps to run the experiment
    :param policy: str representing the policy to use ('random', 'greedy', 'exploit')
    """
    total_rewards = [0] * len(agents)
    step_counts = [0] * len(agents)
    total_distances = [0] * len(agents)
    success_counts = [0] * len(agents)

    # Store the initial Q-tables
    initial_q_tables = [agent.q_table.copy() for agent in agents]

    for step in range(num_steps):
        if step < 500:
            policy = 'random'
        elif policy == 'greedy':
            policy = 'greedy'
        elif policy == 'exploit':
            if np.random.random() < 0.8:
                policy = 'greedy'
            else:
                policy = 'random'

        for i, agent in enumerate(agents):
            available_actions = environment.get_available_actions(agent)

            if 'pickup' in available_actions:
                action = 'pickup'
            elif 'dropoff' in available_actions:
                action = 'dropoff'
            else:
                if policy == 'random':
                    action = agent.choose_random_action(available_actions)
                elif policy == 'greedy':
                    action = agent.choose_best_action(environment.get_state(agent), available_actions)
                elif policy == 'exploit':
                    if np.random.random() < 0.8:
                        action = agent.choose_best_action(environment.get_state(agent), available_actions)
                    else:
                        action = agent.choose_random_action(available_actions)

            prev_position = agent.position
            reward = environment.execute_action(agent, action)
            total_rewards[i] += reward

            if reward > 0:
                success_counts[i] += 1
            
            # Calculate the distance traveled by the agent
            distance = calculate_distance(prev_position, agent.position)
            total_distances[i] += distance

            # Update the Q-table
            state = environment.get_state(agent)
            next_state = environment.get_state(agent)
            agent.update_q_table(state, action, reward, next_state)

        for i in range(len(agents)):
            step_counts[i] = step + 1

        if environment.is_terminal_state():
            environment.reset()

        if (step + 1) % 1000 == 0:
            print(f"Steps completed: {step + 1}")
            for i in range(len(agents)):
                print(f"Agent {i + 1}:")
                print(f"  Average reward over {step + 1} steps: {total_rewards[i] / (step + 1):.3f}")
                print(f"  Success rate over {step + 1} steps: {success_counts[i] / (step + 1) * 100:.2f}%")
                print(f"  Manhattan distance: {total_distances[i]}")
            print()

    print("Final results:")
    for i in range(len(agents)):
        print(f"Agent {i + 1}:")
        print(f"  Total reward over {num_steps} steps: {total_rewards[i]:.3f}")
        print(f"  Total success rate over {num_steps} steps: {success_counts[i] / num_steps * 100:.2f}%")
        print(f"  Total Manhattan distance: {total_distances[i]}")
    print()

    # Print the initial and final Q-tables
    print_q_tables(agents)

    return total_rewards, step_counts, total_distances, success_counts


def main():
    # Define the environment and agents
    grid_size = (5, 5)
    pickup_locations = [(0, 4), (1, 3), (4, 1)]
    dropoff_locations = [(0, 0), (2, 0), (3, 4)]
    environment = Environment(grid_size, pickup_locations, dropoff_locations)

    agents = [
        Agent((0, 2), learning_rate=0.3, discount_factor=0.5, exploration_rate=0.2),
        Agent((2, 2), learning_rate=0.3, discount_factor=0.5, exploration_rate=0.2),
        Agent((4, 2), learning_rate=0.3, discount_factor=0.5, exploration_rate=0.2)
    ]

    for agent in agents:
        environment.add_agent(agent)

    # Visualize the environment
    environment.visualize()

    num_steps = 9000

    print("Experiment 1a:")
    run_experiment(environment, agents, num_steps, policy='random')

    print("Experiment 1b:")
    # run_experiment(environment, agents, num_steps, policy='greedy')

    print("Experiment 1c:")
    # run_experiment(environment, agents, num_steps, policy='exploit')

    # ADD REST OF EXPERIMENTS
    # Experiment 2
    # Experiment 3
    # Experiment 4

if __name__ == "__main__":
    main()