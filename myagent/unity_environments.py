import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod

from unityagents import UnityEnvironment

from .environments import (EnvironmentMgr,
                           EnvironmentNotLoadedError,
                           EnvironmentResetError)


class UnityEnvironmentMgr(EnvironmentMgr):
    def __init__(self, file, start=False):
        self.file = file
        
        self.env = None
        self.brain_name = None
        self.action_size = None
        self.state_size = None
        
        if start:
            self.start()
        
    def __enter__(self):
        return self.start()
    
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()

    def reset(self):
        if self.env is None or self.brain_name is None:
            raise EnvironmentNotLoadedError(
                'Environment Not Initialized, run start method'
            )
        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state
    
    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0] 
        
        return (next_state, reward, done, env_info)
    
    def start(self):
        if self.env is None:
            self.env = self.get_env(self.file)
            self.brain_name = self.env.brain_names[0]
            
            brain = self.env.brains[self.brain_name]
            self.action_size = brain.vector_action_space_size
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            self.state_size = len(state)
        return self.env
    
    def close(self):
        self.env.close()
        raise EnvirontmentResetError(
            "Must Reset Kernel - due to bug in UnityAgents"
        )
    
    @staticmethod
    def get_env(file):
        return UnityEnvironment(file_name=file)

    
class UnityEnvironmentMultiAgentMgr(UnityEnvironmentMgr):
    def __init__(self, file, start=False):
        super().__init__(file, start=start)
        self.n_agents = None
        self.scores = None
        
    def reset(self):
        if self.env is None or self.brain_name is None:
            raise EnvironmentNotLoadedError(
                'Environment Not Initialized, run start method'
            )
        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
#         scores = np.zeros(self.n_agents) -> trainer
        return states
    
    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        
        return (next_states, rewards, dones, env_info)
    
    def start(self):
        if self.env is None:
            self.env = self.get_env(self.file)
            self.brain_name = self.env.brain_names[0]
            
            brain = self.env.brains[self.brain_name]
            self.action_size = brain.vector_action_space_size
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.state_size = states.shape[1]
            self.n_agents = states.shape[0]
        return self.env
    
    @staticmethod
    def get_env(file):
        return UnityEnvironment(file_name=file)