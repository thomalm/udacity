import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import agent

sns.set()


def visualize_run(s, e, r, t, n_rows, row):
    """
    Visualize the percentage of successful trials, rewards per trial and number of negative reward actions

    :param s: the result of each trial (0 failed, 1 success)
    :param e: number of negative reward actions taken per trial
    :param r: total reward per trial
    :param t: number of trials
    :param n_rows: number of experiments
    :param row: experiment number
    :return:
    """

    # Visualize the success rate
    plt.subplot2grid((n_rows, 3), (row, 0))
    labels = ['Success', 'Failure']
    y_pos = np.arange(len(labels))
    res = [sum(s), t - sum(s)]
    barlist = plt.barh(y_pos, res, align='center', alpha=0.5)
    barlist[1].set_color('r')  # color the failure bar red
    plt.yticks(y_pos, labels)
    plt.title("Success rate {} %".format((sum(s, 0.0) / t) * 100));

    # Visualize the rewards
    plt.subplot2grid((n_rows, 3), (row, 1))
    plt.plot(r, 'c')
    plt.title("Total Reward per trial")

    # Visualize the negative reward actions
    plt.subplot2grid((n_rows, 3), (row, 2))
    plt.plot(e, 'r')
    plt.title("Negative reward actions taken per trial");


def visualize_trials(s, t):
    """
    Visualize the trials as a scatter-plot coloring the dots based on the outcome
    """

    plt.scatter(range(1, t + 1), [1] * t, s=50, c=s, cmap='RdYlGn')
    plt.yticks(np.arange(0), []);
    plt.title("Results");
    plt.xlim(0, t + 1);


def run_experiment(alpha, epsilon, gamma, selection, n_runs, trials, f=False):
    """
    Perform n_runs experiments and visualize the results
    """

    stats = []

    # run n_trial experiments
    for i in range(n_runs):
        # Run simulation
        s, m, r = agent.run(alpha, epsilon, gamma, selection, trials, f)
        stats.append(sum(s))
        # Visualize results
        visualize_run(s, m, r, trials, n_runs, i)

    print "Average success rate {0:.3f} %".format((np.mean(stats) / trials) * 100)
