from multiprocess import Pool
from typing import Callable, Collection
import numpy as np
import gym


class Parallel_Environment():
    def __init__(self,
                 env_name: str,
                 reward_function: Callable[[gym.core.Env, np.array, int], float],
                 viz_function: Callable[[gym.core.Env, np.array], float]
                 ):
        self.env_name = env_name
        self.reward_function = reward_function
        self.viz_function = viz_function
        self.observation_space_shape = gym.make(
            env_name).observation_space.shape

    def collect_parallel(self, params: Collection[np.array]) -> np.array:
        pool = Pool()
        seed = np.random.randint(0, 999999)
        try:
            rewards = pool.starmap(
                self.reward_function, [(gym.make(self.env_name), param, seed) for param in params])
            pool.close()
            return np.array(rewards).flatten()
        except KeyboardInterrupt:
            pool.terminate()
        except Exception as e:
            print(e)
            pool.terminate()
        finally:
            pool.close()

    def vizualize(self, params: np.array) -> float:
        reward = self.viz_function(gym.make(self.env_name), params)
        return reward
