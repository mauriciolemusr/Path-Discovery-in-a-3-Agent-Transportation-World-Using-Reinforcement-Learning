import numpy as np

class Agent:
    def __init__(self, position, learning_rate, discount_factor, exploration_rate):
        """
        Initializes the agent.
        :param position: tuple (x, y) representing the initial position of the agent
        :param learning_rate: float representing the learning rate (alpha)
        :param discount_factor: float representing the discount factor (gamma)
        :param exploration_rate: float representing the exploration rate (epsilon)
        """
        self.position = position
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.carrying_block = False

    