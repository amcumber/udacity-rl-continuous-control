import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .agents import Agent, MultiAgent
from .environments import EnvironmentMgr


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class MultiAgentTrainer(Trainer):
    def __init__(
        self,
        agents: MultiAgent,
        env: EnvironmentMgr,
        n_episodes: int = 2000,
        max_t: int = 10000,
        window_len: int = 100,
        solved: float = 30.0,
        save_root: str = "checkpoint",
    ):
        """
        Multi Agent Trainer - Repurposed from DQNTrainer from Project 1.

        Parameters
        ----------
        agent : Agent
            agent to act upon
        env : UnityEnvironmentMgr
            environment manager containing enter and exit methods to call
            UnityEnvironment
            - DO NOT CLOSE in v0.4.0 - this will cause you to be locked
            out of your environment... NOTE TO UDACITY STAFF - fix this issue
            by upgrading UnityEnvironemnt requirements. See
            https://github.com/Unity-Technologies/ml-agents/issues/1167
        n_episodes: int
            maximum number of training episodes
        max_t: int
            maximum number of timesteps per episode
        window_len : int (100)
            update terminal with information for every specified iteration,
        solved : float
            score to be considered solved
        save_file: str
            file to save network weights
        """
        self.agents = agents
        self.env = env
        self.n_episodes = n_episodes
        self.max_t = max_t

        self.solved = solved
        self.window_len = window_len

        self.all_scores_ = None
        self.n_workers = len(self.agents)
        self.root = save_root

    def _report_score(self, i_episode, scores_window, end="") -> None:
        print(
            f"\rEpisode {i_episode+1:d}"
            f"\tAverage Score: {np.mean(scores_window):.2f}",
            end=end,
        )

    def _check_solved(self, i_episode, scores_window) -> None:
        if np.mean(scores_window) >= self.solved:
            print(
                f"\nEnvironment solved in {i_episode+1:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.2f}"
            )
            return True
        return False

    def _get_save_file(self, root):
        now = datetime.now()
        return f'{root}-{now.strftime("%Y%m%dT%H%M%S")}'

    def _run_episode(
        self, all_scores, scores_window, max_t, train_mode=True
    ) -> Tuple[list, deque, float]:
        """Run an episode of the training sequence"""
        states = self.env.reset(train_mode=train_mode)
        new_scores = np.zeros(self.n_workers)
        for _ in range(max_t):
            actions = self.agents.act(states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self.agents.step(states, actions, rewards, next_states, dones)
            states = next_states
            new_scores += rewards
            if np.any(dones):
                break
        scores_window.append(new_scores)  # save most recent score
        all_scores.append(new_scores)  # save most recent score
        return (all_scores, scores_window)

    def train(self):
        self.env.start()
        all_scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.window_len)
        for i_episode in range(self.n_episodes):
            (all_scores, scores_window) = self._run_episode(
                all_scores, scores_window, self.max_t
            )
            self.all_scores_ = all_scores
            self._report_score(i_episode, scores_window)
            if (i_episode + 1) % self.window_len == 0:
                self._report_score(i_episode, scores_window, end="\n")
                self.agents.save(f"{self.root}-checkpoint")
            if self._check_solved(i_episode, scores_window):
                self.agents.save(self._get_save_file(f"{self.root}-solved"))
                break
        return all_scores

    def eval(self, n_episodes=3, t_max=1000):
        ## scores_window
        all_scores = []
        scores_window = deque(maxlen=self.window_len)
        for i in range(n_episodes):
            (all_scores, scores_window) = self._run_episode(
                all_scores, scores_window, t_max, train_mode=False
            )
            self.all_scores_ = all_scores
            print(f"\rEpisode {i+1}\tFinal Score {np.mean(all_scores):.2f}", end="")
        return all_scores
