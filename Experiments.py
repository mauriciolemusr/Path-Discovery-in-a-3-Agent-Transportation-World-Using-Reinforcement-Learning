import numpy as np

class Experiments:
    def run_experiment1(self, environment, agents, num_steps, policy, initial_agent_positions):
        learning_rate = agents[0].learning_rate
        discount_factor = agents[0].discount_factor
        total_rewards = [0] * len(agents)
        total_successes = [0] * len(agents)
        total_distances = [0] * len(agents)

        environment.reset(initial_agent_positions)

        for step in range(num_steps):
            # Switching between exploration and exploitation phases based on step count.
            if step < 500:
                policy_to_use = 'random'
            else:
                policy_to_use = policy

            # Iterating over agents and executing actions.
            for i, agent in enumerate(agents):
                state = environment.get_state(agent)
                action = agent.choose_action(environment, policy_to_use)
                reward = environment.execute_action(agent, action)
                next_state = environment.get_state(agent)

                # Updating total rewards, successes, and distances.
                total_rewards[i] += reward
                if reward > 0:
                    total_successes[i] += 1

                distance = self.calculate_distance(state[0], next_state[0])
                total_distances[i] += distance

                # Updating the Q-table based on the observed state-action-reward transition.
                agent.update_q_table(state, action, reward, next_state, learning_rate)

            # Resetting the environment if it reaches a terminal state.
            if environment.is_terminal_state():
                environment.reset(initial_agent_positions)

        return total_rewards, total_distances, total_successes

    def run_experiment1a(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'random', initial_agent_positions)

    def run_experiment1b(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'greedy', initial_agent_positions)

    def run_experiment1c(self, environment, agents, num_steps, initial_agent_positions):
        return self.run_experiment1(environment, agents, num_steps, 'exploit', initial_agent_positions)

    def run_experiment2(self, environment, agents, num_steps, initial_agent_positions):
        learning_rate = agents[0].learning_rate
        discount_factor = agents[0].discount_factor
        total_rewards = [0] * len(agents)
        total_successes = [0] * len(agents)
        total_distances = [0] * len(agents)

        environment.reset(initial_agent_positions)

        # Initialize state and action for each agent
        states = []
        actions = []
        for agent in agents:
            state = environment.get_state(agent)
            action = agent.choose_action(environment, 'random')
            states.append(state)
            actions.append(action)

        for step in range(num_steps):
            # Switching between exploration and exploitation phases based on step count.
            if step < 500:
                policy_to_use = 'random'
            else:
                policy_to_use = 'exploit'

            # Iterating over agents and executing actions.
            for i, agent in enumerate(agents):
                state = states[i]
                action = actions[i]
                reward = environment.execute_action(agent, action)
                next_state = environment.get_state(agent)

                # Choose next action using the specified policy
                next_action = agent.choose_action(environment, policy_to_use)

                # Updating total rewards, successes, and distances.
                total_rewards[i] += reward
                if reward > 0:
                    total_successes[i] += 1

                distance = self.calculate_distance(state[0], next_state[0])
                total_distances[i] += distance

                # Updating the Q-table based on the observed state-action-reward-next_state-next_action transition.
                agent.update_q_table_sarsa(state, action, reward, next_state, next_action)

                # Update state and action for the next iteration
                states[i] = next_state
                actions[i] = next_action

            # Resetting the environment if it reaches a terminal state.
            if environment.is_terminal_state():
                environment.reset(initial_agent_positions)
                # Reset state and action for each agent
                states = []
                actions = []
                for agent in agents:
                    state = environment.get_state(agent)
                    action = agent.choose_action(environment, 'random')
                    states.append(state)
                    actions.append(action)

        return total_rewards, total_distances, total_successes

    def run_experiment3(self, environment, agents, num_steps, initial_agent_positions, learning_rate):
        discount_factor = agents[0].discount_factor
        total_rewards = [0] * len(agents)
        total_successes = [0] * len(agents)
        total_distances = [0] * len(agents)

        environment.reset(initial_agent_positions)

        for step in range(num_steps):
            # Switching between exploration and exploitation phases based on step count.
            if step < 500:
                policy_to_use = 'random'
            else:
                policy_to_use = 'exploit'

            # Iterating over agents and executing actions.
            for i, agent in enumerate(agents):
                state = environment.get_state(agent)
                action = agent.choose_action(environment, policy_to_use)
                reward = environment.execute_action(agent, action)
                next_state = environment.get_state(agent)

                # Updating total rewards, successes, and distances.
                total_rewards[i] += reward
                if reward > 0:
                    total_successes[i] += 1

                distance = self.calculate_distance(state[0], next_state[0])
                total_distances[i] += distance

                # Updating the Q-table based on the observed state-action-reward transition.
                agent.update_q_table(state, action, reward, next_state, learning_rate)

            # Resetting the environment if it reaches a terminal state.
            if environment.is_terminal_state():
                environment.reset(initial_agent_positions)

        return total_rewards, total_distances, total_successes

    def run_experiment4(self, environment, agents, num_steps, initial_agent_positions):
        learning_rate = agents[0].learning_rate
        discount_factor = agents[0].discount_factor
        total_rewards_before = [0] * len(agents)
        total_successes_before = [0] * len(agents)
        total_distances_before = [0] * len(agents)
        total_steps_before = 0

        total_rewards_after = [0] * len(agents)
        total_successes_after = [0] * len(agents)
        total_distances_after = [0] * len(agents)
        total_steps_after = 0

        terminal_state_count = 0
        environment.reset(initial_agent_positions)

        for step in range(num_steps):
            # Switching between exploration and exploitation phases based on step count.
            if step < 500:
                policy_to_use = 'random'
            else:
                policy_to_use = 'exploit'

            # Iterating over agents and executing actions.
            for i, agent in enumerate(agents):
                state = environment.get_state(agent)
                action = agent.choose_action(environment, policy_to_use)
                reward = environment.execute_action(agent, action)
                next_state = environment.get_state(agent)

                # Updating total rewards, successes, and distances.
                if terminal_state_count < 3:
                    total_rewards_before[i] += reward
                    if reward > 0:
                        total_successes_before[i] += 1
                    distance = self.calculate_distance(state[0], next_state[0])
                    total_distances_before[i] += distance
                else:
                    total_rewards_after[i] += reward
                    if reward > 0:
                        total_successes_after[i] += 1
                    distance = self.calculate_distance(state[0], next_state[0])
                    total_distances_after[i] += distance

                # Updating the Q-table based on the observed state-action-reward transition.
                agent.update_q_table(state, action, reward, next_state, learning_rate)

            # Updating step counts
            if terminal_state_count < 3:
                total_steps_before += 1
            else:
                total_steps_after += 1

            # Resetting the environment if it reaches a terminal state.
            if environment.is_terminal_state():
                terminal_state_count += 1
                environment.reset(initial_agent_positions)

                if terminal_state_count == 3:
                    # Changing pickup locations after reaching the terminal state for the third time.
                    environment.pickup_locations = [(4, 2), (3, 3), (2, 4)]

                if terminal_state_count == 6:
                    break

        return (
            total_rewards_before, total_distances_before, total_successes_before, total_steps_before,
            total_rewards_after, total_distances_after, total_successes_after, total_steps_after,
        )

    def calculate_distance(self, position1, position2):
        return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])