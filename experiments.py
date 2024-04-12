import numpy as np

class Experiments:
    def run_experiment1(self, environment, agents, num_steps, policy, initial_agent_positions):
        total_rewards = [0] * len(agents)
        total_successes = [0] * len(agents)
        total_distances = [0] * len(agents)

        environment.reset(initial_agent_positions)

        for step in range(num_steps):
            if step < 500:
                policy_to_use = 'random'
            else:
                policy_to_use = policy

            for i, agent in enumerate(agents):
                state = environment.get_state(agent)
                action = agent.choose_action(environment, policy_to_use)
                reward = environment.execute_action(agent, action)
                next_state = environment.get_state(agent)

                total_rewards[i] += reward
                if reward > 0:
                    total_successes[i] += 1

                distance = self.calculate_distance(state[0], next_state[0])
                total_distances[i] += distance

                agent.update_q_table(state, action, reward, next_state)

            if environment.is_terminal_state():
                environment.reset(initial_agent_positions)

        return total_rewards, total_distances, total_successes

    def run_experiment1a(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'random', initial_agent_positions)

    def run_experiment1b(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'greedy', initial_agent_positions)

    def run_experiment1c(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'exploit', initial_agent_positions)

    def calculate_distance(self, position1, position2):
        return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])