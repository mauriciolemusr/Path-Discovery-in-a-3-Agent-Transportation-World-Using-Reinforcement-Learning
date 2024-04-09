import numpy as np
from environment import Environment
from agent import Agent

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

    #Visualize the environment
    environment.visualize()
if __name__ == "__main__":
    main()