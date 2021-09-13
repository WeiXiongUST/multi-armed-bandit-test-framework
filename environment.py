import numpy as np
import matplotlib.pyplot as plt


class environment():
    def __init__(self, bandits, agents):
        self.bandits = bandits
        self.agents = agents
        self.results = None
        self.K = len(self.bandits)
        self.M = len(self.agents)

    def reset(self):
        for i in range(self.M):
            self.agents[i].reset()

    def run(self, horizon=10000, experiments=1):
        results = np.zeros((self.M, experiments, horizon))
        for m in range(self.M):
            agent = self.agents[m]
            for i in range(experiments):
                self.reset()
                for t in range(horizon):
                    action = agent.select_arm()
                    reward = self.bandits[action].draw()
                    results[m][i][t] = reward
                    agent.update(action, reward)

        self.results = results

    def plot_result(self, result, ax):
        horizon = result.shape[1]
        top_mean = self.bandits[0].mean_return
        for i in range(1, self.K):
            if self.bandits[i].mean_return > top_mean:
                top_mean = self.bandits[i].mean_return
        best_case_reward = top_mean * np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(result, axis=1)
        regret = best_case_reward - cumulated_reward[:horizon]

        y = np.mean(regret, axis=0)
        x = np.arange(len(y))
        std = np.std(regret, axis=0)
        #print(len(std))

        y_up_err = y + std
        y_low_err = y - std
        ax.plot(x, y)
        ax.fill_between(x, y_low_err, y_up_err, alpha=0.3)
        #plt.show()

    def plot_results(self):
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots()
        for m in range(self.M):
            result = self.results[m]
            self.plot_result(result, ax)
        plt.ylim(-100, 7300)
        plt.legend([self.agents[i].get_name() for i in range(self.M)])
        plt.xlabel("Time step")
        plt.ylabel("Regret")
        plt.show()
