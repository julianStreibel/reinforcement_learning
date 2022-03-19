import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from .Parallel_Environment import Parallel_Environment


class Success_Matching():
    def __init__(self, n_states):
        self.n_states = n_states
        self.mean = np.random.rand(self.n_states)
        self.cov = np.eye(self.n_states)

    def train(
        self,
        env: Parallel_Environment,
        epochs=20,
        population_size=10,
        variance_optimization=True,
        variance_decay=False,
        lower_bound_variance=0,
        viz=False,
        verbose=True,
        lower_bound_importance=1e-4
    ):
        with tqdm(total=epochs, disable=not verbose) as pbar:
            for epoch in range(epochs):
                # sample params
                samples = np.random.multivariate_normal(
                    self.mean, self.cov, size=(population_size))

                # collect rewards
                reward_sums = env.collect_parallel(samples)

                # build success weights
                reward_sums_max = reward_sums.max()
                beta = 10 / (reward_sums_max - reward_sums.min() + 1e-6)
                w = np.exp(beta * (reward_sums - reward_sums_max))
                w[w < lower_bound_importance] = 0

                # update the distribution
                new_mean = w @ samples / w.sum()
                if variance_optimization:
                    distance_to_mean = (samples - new_mean)
                    new_cov = distance_to_mean.T @ np.diag(
                        w) @ distance_to_mean / w.sum()
                if variance_decay:
                    new_cov = np.identity(self.n_states) / np.log(epoch + 2)
                new_var = np.maximum(np.diag(new_cov), lower_bound_variance)
                np.fill_diagonal(new_cov,  new_var)
                self.mean = new_mean
                self.cov = new_cov
                pbar.update(1)
                pbar.set_postfix({
                    "max reward": reward_sums.max(),
                    "mean reward": reward_sums.mean(),
                    "var reward": reward_sums.var()
                })

                if viz:
                    env.vizualize(self.mean)
            return self.mean
