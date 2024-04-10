import numpy as np

class Agent:
    def __init__(self, position, learning_rate, discount_factor, exploration_rate):
            """
            Initializes the agent.
            tuple (x, y) representing the initial position of the agent
            float representing the learning rate (alpha)
            float representing the discount factor (gamma)
            float representing the exploration rate (epsilon)
            """
            self.position = position
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.exploration_rate = exploration_rate
            self.q_table = {}
            self.carrying_block = False



    def choose_action(self, environment):
        """
        Chooses an action based on the current state and exploration rate.
        :param environment: Environment object
        :return: str representing the chosen action
        """
        state = environment.get_state(self)
        available_actions = environment.get_available_actions(self)

        if 'pickup' in available_actions:
            return 'pickup'
        elif 'dropoff' in available_actions:
            return 'dropoff'
        else:
            if np.random.random() < self.exploration_rate:
                return self.choose_random_action(available_actions)
            else:
                return self.choose_best_action(state, available_actions)



    def choose_random_action(self, available_actions):
        """
        Chooses a random action from the available actions.
        :param available_actions: list of str representing the available actions
        :return: str representing the chosen action
        """
        return np.random.choice(np.array(available_actions))



    def choose_best_action(self, state, available_actions):
        """
        Chooses the action with the highest Q-value for the current state.
        :param state: tuple representing the current state
        :param available_actions: list of str representing the available actions
        :return: str representing the chosen action
         """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        q_values = self.q_table[state]
        max_q_value = max(q_values.values())
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q_value]
        return np.random.choice(best_actions)


    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the observed transition and reward.
        :param state: tuple representing the current state
        :param action: str representing the action taken
        :param reward: float representing the reward received
        :param next_state: tuple representing the next state
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ['up', 'down', 'left', 'right', 'pickup', 'dropoff']}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value



    def reset(self):
        """
        Resets the agent to its initial state.
        """
        self.position = (0, 0)
        self.carrying_block = False