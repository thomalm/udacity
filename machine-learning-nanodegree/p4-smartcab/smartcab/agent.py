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

    1. Initialize Q(s, a) arbitrarily
    2. For each trial t:
        3. Repeat for until deadline or goal is reached:
            4. Update the state s
            5. Choose action a from s using a policy derived from Q (e.g. epsilon-greedy)
            6. Take action a and observe the reward and outcome state s'
            7. Update Q(s, a) := Q(s,a) + α [r + γ max_a′ Q(s′,a′) − Q(s,a)]
    """

    def __init__(self, env):
        """ Initialize the learning agent"""

        super(LearningAgent, self).__init__(env)  # Run constructor of base class
        self.color = 'red'  # Set agent color to red, overriding the default value
        self.planner = RoutePlanner(self.env, self)  # Route planner to get next_waypoint
        self.actions = Environment.valid_actions  # Store all possible actions (improved code readability)

        self.Q = defaultdict(lambda: random.uniform(-.25, .25))  # arbitrary values for all state, action pairs
        self.gamma = 0.35  # Discount factor of max(Qs', a')
        self.epsilon = 0.9  # Probability of doing a random move
        self.alpha = 0.2  # Learning rate
        self.trial = 0  # Trial counter (epsilon decay)
        self.selection = "random"
        self.filter = False

        self.total_reward = 0  # Keep track of total reward for the current trial
        self.mistake_counter = 0  # Keep track of the number of moves with a negative reward
        self.rewards = []  # Historical rewards
        self.mistakes = []  # Historical mistakes
        self.results = []  # Non-successful trials
        self.success = 1

    def set_parameters(self, alpha, epsilon, gamma, selection, f):
        """ Overwrite the parameters with static settings hack """

        self.gamma = gamma  # Discount factor of max(Qs', a')
        self.epsilon = epsilon  # Probability of doing a random move
        self.alpha = alpha
        self.selection = selection
        self.filter = f

    def reset(self, destination=None):
        """ Reset the agent between trails """

        self.planner.route_to(destination)  # Initialize new destination

        self.trial += 1  # For every trial, increment trial counter
        self.rewards.append(self.total_reward)  # Keep track of rewards for each trial
        self.mistakes.append(self.mistake_counter)  # Keep track of mistakes for each trial
        self.total_reward = 0  # Reset reward counter
        self.mistake_counter = 0  # Reset mistake counter
        self.results.append(self.success)
        self.success = 1

    def get_state(self):
        """ Return the current state s of the agent """

        inputs = self.env.sense(self)
        return inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint

    def select_action(self, s):

        if self.selection == "random":
            return random.choice(self.actions)

        if self.selection == "epsilon-greedy":
            # select a random move with probability ε
            if random.random() < self.epsilon:
                return random.choice(self.actions)
            else:
                # Shuffle deals with the cases when a draw is returned from np.argmax
                random.shuffle(self.actions)
                # Evaluate all action and pick the one with the highest estimated reward
                return self.actions[np.argmax([self.Q[(s, a_i)] for a_i in self.actions])]

        elif self.selection == "epsilon-decay":
            # select a random move with probability ε controlled by an epsilon-decay function
            if random.random() < np.exp(-self.epsilon * self.trial):
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

        # Filter out unwanted actions
        if self.filter:
            self.actions = [self.next_waypoint, None]

        # Observe the current state
        s = self.get_state()
        # Select action a according to policy
        a = self.select_action(s)
        # Take action a, observe reward and s'
        reward, s_i = self.env.act(self, a), self.get_state()
        # Calculate maximum attainable reward in the next state (s_i)
        max_a = max([self.Q[(s_i, a_i)] for a_i in self.actions])
        # Update state, action value
        self.Q[(s, a)] += self.alpha * (reward + self.gamma * (max_a - self.Q[(s, a)]))

        # Update total reward
        self.total_reward += reward
        # If remaining time is 0, store as failed attempt
        if self.env.get_deadline(self) == 0:
            self.success = 0
        # Update total mistakes
        if reward < 0:
            self.mistake_counter += 1

        # DEBUG
        # print "deadline = {}, inputs = {}, action = {}, reward = {}".format(
            # self.env.sense(self), self.env.get_deadline(self), a, reward)

    def get_run_data(self):
        return self.results, self.mistakes, self.rewards


def run(alpha, epsilon, gamma, selection, t, f):
    """Run the agent for a finite number of trials."""

    # create environment, add dummy agents
    e = Environment()

    # Create our agent
    a = e.create_agent(LearningAgent)
    # Initialize agent settings
    a.set_parameters(alpha, epsilon, gamma, selection, f)
    # Configure the environment, set agent a as primary agent to track
    e.set_primary_agent(a, enforce_deadline=True)

    # Simulator settings, reduce delay to speed up simulations
    sim = Simulator(e, update_delay=0.00001)
    # Set the number of trials
    sim.run(n_trials=t)

    # Return data on agents performance
    return a.get_run_data()


if __name__ == '__main__':
    run(.2, .5, .1, "epsilon-decay", 100)
