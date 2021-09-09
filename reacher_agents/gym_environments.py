from typing import Tuple
import gym

from .environments import EnvironmentMgr, EnvironmentNotLoadedError


# Classes
class GymEnvironmentMgr(EnvironmentMgr):
    def __init__(self, scenario, seed=42):
        """Initialize an environment manager with a given gym scenario"""
        self.scenario = scenario
        self.seed = seed

        self.env = None
        self.action_size = None
        self.state_size = None

    def __enter__(self):
        return self.start()

    def __exit__(self, e_type, e_value, e_traceback):
        self.env.close()

    def step(self, action) -> Tuple["next_state", "reward", "done", "env_info"]:
        """Take a step within the given environment"""
        return self.env.step(action)

    def reset(self) -> "state":
        """Reset the state of the environment"""
        if self.env is None:
            raise EnvironmentNotLoadedError(
                "Environment Not Initialized, run start method"
            )
        return self.env.reset()

    def start(self):
        """Start the loaded environment"""
        if self.env is None:
            self.env = self.get_env(self.scenario)
            self.env.seed(self.seed)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.n
        return self.env

    @staticmethod
    def get_env(scenario):
        return gym.make(scenario)

    def close(self):
        self.env.close()
