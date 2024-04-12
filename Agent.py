import numpy as np

class Agent:
    def __init__(self, initial_position, learning_rate, discount_factor):
        # Initializing the Agent class with its initial position, learning rate, discount factor, an empty Q-table, and not carrying any block.
        self.position = initial_position
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.carrying_block = False

    # Resetting the agent's state by setting it to not carrying any block.
    def reset(self):
        self.carrying_block = False

    # Choosing an action for the agent based on a given policy.
    def choose_action(self, environment, policy):
        state = environment.get_state(self)  # Get the current state of the agent from the environment.
        available_actions = environment.get_available_actions(self)  # Get the available actions for the agent.

        # If the current state is not in the Q-table, initialize it with zero Q-values for each available action.
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}

        # Actions based on the specified policy.
        if policy == 'random':  # PRANDOM
            return np.random.choice(available_actions)
        elif policy == 'greedy':  # PGREEDY
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            return np.random.choice(best_actions)  # Randomly choose among the best actions (in case of ties).
        elif policy == 'exploit':  # PEXPLOIT
            if np.random.random() < 0.8:  # With 80% probability, exploit the best action.
                max_q_value = max(self.q_table[state].values())
                best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
                return np.random.choice(best_actions)  # Randomly choose among the best actions (in case of ties).
            else:  # With 20% probability, explore by choosing randomly from available actions.
                return np.random.choice(available_actions)

           
    # Updating the Q-table based on the state, action, reward, and the next state.
    def update_q_table(self, state, action, reward, next_state, learning_rate):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}

        old_value = self.q_table[state].get(action, 0.0)  # Retrieving the old Q-value, defaulting to 0.0 if not found
        next_max = max(self.q_table[next_state].values())  # Finding the maximum Q-value for the next state

        # Updating the Q-value using the Q-learning algorithm
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value  # Assigning the new Q-value to the Q-table

    def update_q_table_sarsa(self, state, action, reward, next_state, next_action):
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}

            old_value = self.q_table[state][action]  # Retrieving the old Q-value
            next_value = self.q_table[next_state][next_action]  # Retrieving the Q-value for the next state and next action

            # Updating the Q-value using the SARSA algorithm
            new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_value)
            self.q_table[state][action] = new_value  # Assigning the new Q-value to the Q-table