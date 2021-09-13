from algorithms import *
from arms import *
from environment import *

K = 10
T = 100000
experiment = 10
bandits = [BernoulliArm(np.random.random()) for i in range(K)]
agent = [UCB1(K, T), UniformExploration(K, T), AnnealingEpsilonGreedy(K, T), ThompsonSampling(K, T)]
MAB = environment(bandits, agents=agent)
MAB.run(T, experiment)
MAB.plot_results()
