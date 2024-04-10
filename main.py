import numpy as np
from environment import Environment
from agent import Agent

def calculate_avg_distance(positions):
        """
        Calculates the average Manhattan distance between all pairs of positions.
        list of tuples (x, y) representing the positions
        float representing the average Manhattan distance
        """
        num_positions = len(positions)
        total_distance = 0

        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                pos1 = positions[i]
                pos2 = positions[j]
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                total_distance += distance

        avg_distance = total_distance / (num_positions * (num_positions - 1) / 2)
        return avg_distance



def print_q_tables(agents, initial_q_tables):
    """
    Prints the initial and final Q-tables for each agent.
    :param agents: list of Agent objects
    :param initial_q_tables: list of dictionaries representing the initial Q-tables for each agent
    """
    for i, agent in enumerate(agents):
        print(f"Agent {i+1}:")
        print("Initial Q-table:")
        print_q_table(initial_q_tables[i])
        print("Final Q-table:")
        print_q_table(agent.q_table)
        print()

def print_q_table(q_table):
    """
    Prints the given Q-table in a readable format.
    :param q_table: dictionary representing the Q-table
    """
    for state, actions in q_table.items():
        print(f"State: {state}")
        for action, q_value in actions.items():
            print(f"  Action: {action}, Q-value: {q_value:.3f}")
        print()




def run_experiment(environment, agents, num_steps, policy):
        """
        Runs an experiment with the given environment, agents, number of steps, and policy.
        Environment object
        list of Agent objects
        int representing the number of steps to run the experiment
        str representing the policy to use ('random', 'greedy', 'exploit')
        """
        total_rewards = []
        total_reward = 0
        step_counts = []
        avg_agent_distances = []

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

            agent_positions = [agent.position for agent in agents]
            avg_distance = calculate_avg_distance(agent_positions)
            avg_agent_distances.append(avg_distance)

            total_rewards.append(total_reward)
            step_counts.append(step)
            avg_agent_distances.append(avg_distance)

            if (step + 1) % 1000 == 0:
                print(f"Steps completed: {step + 1}")
                print(f"Total reward: {total_reward}")
                print(f"Average agent distance: {avg_distance}")
                print()

            for agent in agents:
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

                        reward = environment.execute_action(agent, action)
                        total_reward += reward

                    if environment.is_terminal_state():
                        step_counts.append(step + 1)
                        environment.reset()

                    if (step + 1) % 1000 == 0:
                        print(f"Steps completed: {step + 1}")

        return total_reward, step_counts, avg_agent_distances

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

        # Store the initial Q-tables
        initial_q_tables = [agent.q_table.copy() for agent in agents]

        #Visualize the environment
        environment.visualize()

        num_steps = 9000

        print("Experiment 1a:")
        run_experiment(environment, agents, num_steps, policy='random')
        print_q_tables(agents, initial_q_tables)

        print("Experiment 1b:")
        #run_experiment(environment, agents, num_steps, policy='greedy')

        print("Experiment 1c:")
        #run_experiment(environment, agents, num_steps, policy='exploit')


        #ADD REST OF EXPERIMENTS
        # Experiment 2
        # Experiment 3
        # Experiment 4

if __name__ == "__main__":
        main()