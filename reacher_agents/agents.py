from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass
        
    @abstractmethod
    def act(self, state, eps=0.):
        pass
    
    @abstractmethod
    def save(self, file):
        pass
    
    @abstractmethod
    def load(self, file):
        pass


class MultiAgent(Agent):...
