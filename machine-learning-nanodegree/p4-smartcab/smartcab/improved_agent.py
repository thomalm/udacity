# coding=utf-8
import random
import numpy as np
from collections import defaultdict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """ An agent that learns to drive in the smart-cab world

    *** Q-learning agent pseudo-code ***

    * Initialize Q(s, a) arbitrarily
    * For each trial t:
        * Repeat for until deadline or goal is reached:
            * Update the state s
            * Choose action a from s using a policy derived from Q (e.g. epsilon-greedy)
            * Take action a and observe the reward and outcome state s'
            * Update Q(s, a) := Q(s,a) + α [r + γ max_a′ Q(s′,a′) − Q(s,a)]
    """

    def __init__(self, env):
        """ Initialize the learning agent"""

        super(LearningAgent, self).__init__(env)  # Run constructor of base class
        self.color = 'red'  # Set agent color to red, overriding the default value
        self.planner = RoutePlanner(self.env, self)  # Route planner to get next_waypoint
        self.actions = Environment.valid_actions  # Store all possible actions (improved code readability)

        self.Q = defaultdict(lambda: random.random())  # Initialize arbitrary values for all state, action pairs
        self.gamma = 0.35  # Discount factor of max(Qs', a')
        self.epsilon = 0.9  # Probability of doing a random move
        self.alpha = 0.2  # Learning rate
        self.trial = 0  # Trial counter (epsilon decay)

        self.total_reward = 0  # Keep track of total reward for the current trial
        self.mistake_counter = 0  # Keep track of the number of moves with a negative reward
        self.rewards = []  # Historical rewards
        self.mistakes = []  # Historical mistakes
        self.results = []  # Non-successful trials

    def reset(self, destination=None):
        """ Reset the agent between trails """

        self.planner.route_to(destination)  # Initialize new destination

        self.trial += 1  # For every trial, increment trial counter
        self.rewards.append(self.total_reward)  # Keep track of rewards for each trial
        self.mistakes.append(self.mistake_counter)  # Keep track of mistakes for each trial
        self.total_reward = 0  # Reset reward counter
        self.mistake_counter = 0  # Reset mistake counter

    def get_state(self):
        """ Return the current state s of the agent """

        inputs = self.env.sense(self)
        return inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint

    def select_action(self, s):
        # select a random move with probability ε controlled by an epsilon-decay function
        if random.random() < self.epsilon / (self.trial + self.epsilon):
            return random.choice(self.actions)
        # otherwise action = max Q(s', a')
        else:
            # Shuffle deals with the cases when a draw is returned from np.argmax
            random.shuffle(self.actions)
            # Evaluate all action and pick the one with the highest estimated reward
            return self.actions[np.argmax([self.Q[(s, a_i)] for a_i in self.actions])]

    def update(self, t):
        """ Implements the q-learning algorithm """

        # Get next waypoint from the route planner
        self.next_waypoint = self.planner.next_waypoint()
        # Observe the remaining time
        deadline = self.env.get_deadline(self)
        # Observe the current state
        s = self.get_state()

        # Select action a according to policy
        a = self.select_action(s)

        # Take action a, observe reward and s'
        reward, s_i = self.env.act(self, a), self.get_state()

        # Calculate maximum attainable reward in the next state (s_i)
        max_a = max([self.Q[(s_i, a_i)] for a_i in self.actions])
        # Update state, action value
        self.Q[(s, a)] += self.alpha * (reward + self.gamma * max_a - self.Q[(s, a)])

        # Update total reward
        self.total_reward += reward

        # Store failed attempts
        if deadline == 0:
            self.results.append(self.trial)

        # Update total mistakes
        if reward < 0:
            self.mistake_counter += 1

        # print "deadline = {}, inputs = {}, action = {}, reward = {}".format(
            # self.env.sense(self), self.env.get_deadline(self), a, reward)

    def get_run_statistics(self):
        return self.results, self.mistakes, self.rewards


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    return a.get_run_statistics()


if __name__ == '__main__':
    run()
