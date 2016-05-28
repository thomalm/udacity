# coding=utf-8
import random
import numpy as np
from collections import defaultdict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):

        # Run constructor of base class
        super(LearningAgent, self).__init__(env)
        # Set agent color to red, overriding the default value
        self.color = 'red'
        # Route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)

        # Initialize arbitrary values for all state, action pairs
        self.Q = defaultdict(lambda: random.random())
        # Memory / discount factor of max(Qs', a')
        self.gamma = 0.35
        # Probability of doing a random move
        self.epsilon = 0.9
        # Learning rate
        self.alpha = 0.2
        # Trial counter (epsilon decay)
        self.trial = 0
        # Store all possible actions (improved code readability)
        self.actions = Environment.valid_actions
        # Keep track of total reward
        self.total_reward = 0
        # Keep track of the historical reward data of the learner
        self.rewards = []

    def reset(self, destination=None):

        # Initialize new destination
        self.planner.route_to(destination)

        # For every trial, update counter
        self.trial += 1
        # Keep track of historic rewards
        self.rewards.append(self.total_reward)
        # Reset reward counter
        self.total_reward = 0

    def get_state(self):
        """ Return the current state s of the agent """
        inputs = self.env.sense(self)
        return inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint

    def update(self, t):
        """ Implements the q-learning algorithm """

        # Get next way-point from the route planner
        self.next_waypoint = self.planner.next_waypoint()

        # Update state
        state = self.get_state()

        # Select action according to policy

        # select a random move with probability ε controlled by an epsilon-decay function
        if random.random() < self.epsilon / (self.trial + self.epsilon):
            action = random.choice(self.actions)
        # otherwise action = max Q(s', a')
        else:
            # Shuffle deals with the cases when a draw is returned from argmax
            random.shuffle(self.actions)
            action = self.actions[np.argmax([self.Q[(state, a)] for a in self.actions])]

        # Take action a, observe reward and s_i
        reward, s_i = self.env.act(self, action), self.get_state()

        # Update total reward
        self.total_reward += reward

        # Calculate maximum attainable reward in the next state (s_i)
        max_a = max([self.Q[(s_i, a)] for a in self.actions])

        # Update Q(s,a) <-- Q(s, a) + α [r + γ max_a Q(s', a') - Q(s, a)]
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max_a - self.Q[(state, action)])

        # [debug]
        # print "deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
