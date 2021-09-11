import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .agents import Agent
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
        agent: Agent,
        env: EnvironmentMgr,
        n_episodes: int,
        max_t: int,
        window_len: int,
        solved: float,
        n_workers: int,
        max_samples: int,
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
        workers: int (1)
            number of workers (1, 20)
        max_sample: int (10)
            max number of workers to sample for multi agent training
        save_file: str
            file to save network weights
        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_t = max_t

        self.solved = solved
        self.window_len = window_len

        self.scores_ = None
        self.n_workers = n_workers
        self.max_samples = max_samples
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

    def train(self):
        self.env.start()
        all_scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.window_len)
        for i_episode in range(self.n_episodes):
            (all_scores, scores_window) = self._run_episode(
                all_scores, scores_window, self.max_t
            )
            self.scores_ = all_scores
            self._report_score(i_episode, scores_window)
            if (i_episode + 1) % self.window_len == 0:
                self._report_score(i_episode, scores_window, end="\n")
                self.agent.save(f"{self.root}-checkpoint")
            if self._check_solved(i_episode, scores_window):
                self.agent.save(self._get_save_file(f"{self.root}-solved"))
                break
        return all_scores

    def _run_episode(
        self, all_scores, scores_window, max_t, render=False
    ) -> Tuple[list, deque, float]:
        """Run an episode of the training sequence"""
        states = self.env.reset()
        self.agent.reset()
        new_scores = np.zeros(self.n_workers)
        for _ in range(max_t):
            if render:
                self.env.render()
            actions = self.agent.act(states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self._step_agents(states, actions, rewards, next_states, dones)
            states = next_states
            new_scores += rewards
            if np.any(dones):
                break
        scores_window.append(new_scores)  # save most recent score
        all_scores.append(new_scores)  # save most recent score
        return (all_scores, scores_window)

    def _step_agents(self, states, actions, rewards, next_states, dones):
        """
        Step Agents depending on number of workers

        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        samples = np.min([self.max_samples, self.n_workers])
        for idx in random.sample(range(self.n_workers), samples):
            self.agent.step(
                states[idx],
                actions[idx],
                rewards[idx],
                next_states[idx],
                dones[idx],
            )

    def eval(self, n_episodes=3, t_max=1000, render=False):
        ## scores_window
        all_scores = []
        scores_window = deque(maxlen=self.window_len)
        for i in range(n_episodes):
            (all_scores, scores_window) = self._run_episode(
                all_scores, scores_window, t_max, render=render
            )
            self.scores_ = all_scores
            print(f"\rEpisode {i+1}\tFinal Score: {np.mean(all_scores):.2f}", end="")
        return all_scores


class SingleAgentTrainer(MultiAgentTrainer):
    def __init__(
        self,
        agent: Agent,
        env: EnvironmentMgr,
        n_episodes: int = 3000,
        max_t: int = 500,
        window_len: int = 100,
        solved: float = 30.0,
        max_samples: int = 10,
        save_root: str = "checkpoint",
        n_workers: int = None
    ):
        super().__init__(
            agent=agent,
            env=env,
            n_episodes=n_episodes,
            max_t=max_t,
            window_len=window_len,
            solved=solved,
            max_samples=None,
            save_root=save_root,
            n_workers=1
        )
    def _step_agents(self, states, actions, rewards, next_states, dones):
        """
        Step Agents depending on number of workers

        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        self.agent.step(
            states,
            actions,
            rewards,
            next_states,
            dones,
        )