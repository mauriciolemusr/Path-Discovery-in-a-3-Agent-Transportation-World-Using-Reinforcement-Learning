import numpy as np

class Agent:
    def __init__(self, initial_position, learning_rate, discount_factor):
        self.position = initial_position
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.carrying_block = False

    def reset(self):
        self.carrying_block = False

    def choose_action(self, environment, policy):
        state = environment.get_state(self)
        available_actions = environment.get_available_actions(self)

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}

        if policy == 'random':
            return np.random.choice(available_actions)
        elif policy == 'greedy':
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            return np.random.choice(best_actions)
        elif policy == 'exploit':
            if np.random.random() < 0.8:
                max_q_value = max(self.q_table[state].values())
                best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
                return np.random.choice(best_actions)
            else:
                return np.random.choice(available_actions)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value