import random


class BernoulliArm():
    def __init__(self, p):
        self.p = p
        self.mean_return = self.p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


class GaussianArm():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.mean_return = self.mu

    def draw(self):
        return random.gauss(self.mu, self.sigma)


class AdversarialArm():
    def __init__(self, reward_sequence):
        self.reward_sequence = reward_sequence
        self.T0 = len(reward_sequence)
        self.step = 0
        self.mean_return = sum(self.reward_sequence) / self.T0

    def draw(self):
        if self.step < self.T0:
            tmp = self.reward_sequence[self.step]
            self.step += 1
        else:
            tmp = random.random

        return tmp

