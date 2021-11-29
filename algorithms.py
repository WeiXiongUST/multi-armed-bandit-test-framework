import numpy as np


class Policy(object):
    def pull(self):
        return 0

    def update(self, play, reward):
        pass

    def get_name(self):
        return "No name"


class UCB1(Policy):
    """
    The UCB1 algorithm will always choose the arm with largest UCB, i.e.,
    sample mean + bonus (confidence radius).
    """
    def __init__(self, narms, horizon):
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        """
        If an arm has never been selected, its UCB is viewed as infinity
        """
        for arm in range(self.K):
            if self.npulls[arm] == 0:
                return arm
        ucb = [0.0 for arm in range(self.K)]
        for arm in range(self.K):
            radius = np.sqrt((2 * np.log(self.t)) / float(self.npulls[arm]))
            ucb[arm] = self.sample_mean[arm] + radius
        return np.argmax(ucb)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

    def reset(self):
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def get_name(self):
        return "UCB1"


class UniformExploration(Policy):
    """
    The Uniform Exploration algorithm first explores each arm N times. Then, it finds the empirically optimal
    arm hat_k which is of the maximal sample mean by the KN samples. Afterward, the algorithm fixes on hat_k
    forever. Here, N is set to be the theoretically optimal value.
    """
    def __init__(self, narms, horizon):
        self.K = narms
        self.T = horizon
        self.N = np.ceil(self.T ** 0.67 * np.log(self.T))
        if self.N > self.T / narms:
            self.N = np.ceil(self.T / narms / 2)
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        for arm in range(self.K):
            if self.npulls[arm] < self.N:
                return arm
        return np.argmax(self.sample_mean)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

    def reset(self):
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def get_name(self):
        return "Uniform Exploration"


class EpsilonGreedy(Policy):
    """
    The Epsilon Greedy algorithm chooses an arm in a randomized fashion:
    1 with probability (1-epsilon), it chooses the arm with largest sample mean;
    2 with probability epsilon, it randomly chooses an arm.
    """
    def __init__(self, narms, horizon, epsilon=0.1):
        self.epsilon = epsilon
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0
        self.count = 0

    def select_arm(self):
        t = self.t + 1
        epsilon = self.epsilon
        if np.random.random() > epsilon:
            return np.argmax(self.sample_mean)
        else:
            self.count += 1
            return np.random.randint(self.K)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

    def reset(self):
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0
        self.count = 0


class AnnealingEpsilonGreedy(Policy):
    """
    Similar to Epsilon Greedy algorithm but the sequence of exploration probability is annealing.
    """
    def __init__(self, narms, horizon, c=0.2, d=0.05):
        self.c = c
        self.d = d
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0
        self.count = 0

    def select_arm(self):
        t = self.t + 1
        epsilon = self.c * self.K / (self.d**2 * t)
        if np.random.random() > epsilon:
            return np.argmax(self.sample_mean)
        else:
            self.count += 1
            return np.random.randint(self.K)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

    def reset(self):
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0
        self.count = 0

    def get_name(self):
        return "Annealing epsilon Greedy"


class RandomExploration(Policy):
    """
    The Random Exploration algorithm randomly chooses an arm (with equal probability 1/K) forever.
    """
    def __init__(self, narms, horizon):
        self.K = narms

    def select_arm(self):
        return np.random.randint(self.K)

    def reset(self):
        pass


class ThompsonSampling(Policy):
    """
    The Thompson Sampling algorithm is a Bayesian algorithm. It maintains K Beta distributions Beta_k(win+1, loss+1)
    where win is the number of samples in which the arm k returns reward 1 and loss is that with reward 0.
    At each step, the algorithm samples K values from K Beta distributions. Then, it chooses the arm with largest value.
    """
    def __init__(self, narms, horizon):
        self.K = narms
        self.T = horizon
        self.win = np.zeros(self.K, dtype=np.int32)
        self.loss = np.zeros(self.K, dtype=np.int32)

    def select_arm(self):
        tmp = np.random.beta(self.win+1, self.loss+1)
        action = np.argmax(tmp)
        return action

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.win[chosen_arm] += 1
        else:
            self.loss[chosen_arm] += 1

    def reset(self):
        self.win = np.zeros(self.K, dtype=np.int32)
        self.loss = np.zeros(self.K, dtype=np.int32)

    def get_name(self):
        return "Thompson Sampling"


class TSFunctionApproximation(Policy):
    def __init__(self, narms, horizon, functions):
        self.K = narms
        self.T = horizon
        self.functions = functions
        self.N = len(self.functions)
        self.likelihood = np.zeros(self.N)

    def select_arm(self):
        prob = np.exp(self.likelihood)
        prob = prob / np.sum(prob)
        choice_function = np.random.choice(self.N, p=prob)
        max_value = self.functions[choice_function].evaluate(0)
        max_idx = 0
        for i in range(1, self.K):
            tmp = self.functions[choice_function].evaluate(i)
            if tmp > max_value:
                max_value = tmp
                max_idx = i
        return max_idx

    def update(self, chosen_arm, reward):
        for i in range(self.N):
            self.likelihood[i] -= (reward - self.functions[i].evaluate(chosen_arm)) ** 2

    def get_name(self):
        return "Thompson Sampling with function approximation"

    
class MaillardSampling(Policy):
    """
    The Maillard sampling algorithm is a Bayesian algorithm. It maintains K Beta distributions Beta_k(win+1, loss+1)
    where win is the number of samples in which the arm k returns reward 1 and loss is that with reward 0.
    At each step, the algorithm samples K values from K Beta distributions. Then, it chooses the arm with largest value.
    See "Maillard Sampling: Boltzmann Exploration Done Optimally" for details.
    """
    def __init__(self, narms, horizon):
        self.K = narms
        self.T = horizon
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        tmp = np.zeros(self.K)
        empirical_opt = np.max(self.sample_mean)
        if self.t < self.K:
            return self.t
        for i in range(self.K):
            tmp[i] = np.exp(-self.npulls[i]/2 * (empirical_opt - self.sample_mean[i]) ** 2)
        tmp = tmp / np.sum(tmp)
        action = np.random.choice(list(range(self.K)), p=tmp)
        return action

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        self.t += 1

    def reset(self):
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.sample_mean = np.zeros(self.K)
        self.t = 0

    def get_name(self):
        return "Maillard Sampling"
