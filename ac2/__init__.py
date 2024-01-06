
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import math
import random

import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from pettingzoo.utils.env import AECEnv, ObsType, ActionType, AgentID
from pettingzoo.utils.conversions import aec_to_parallel

import gymnasium as gym


# Constants.

AGENT_CHANNEL = 0
OBSTACLE_CHANNEL = 1
PHEROMONE_CHANNEL = 2

ACTION_NORTH = 0
ACTION_EAST = 1
ACTION_SOUTH = 2
ACTION_WEST = 3


# Helpers.

def PPL(x: np.ndarray) -> float:
    """Compute the perplexity."""

    distribution = (x + 1e-8) / (x + 1e-8).sum()
    entropy = -(np.log(distribution) * distribution).sum()
    perplexity = np.exp(entropy)

    return perplexity

def normalize(x: np.array):
    return (x - x.min()) / (x.max() - x.min())


@dataclass(frozen=True)
class AC2Configuration:
    """AC2 configuration.

    Attributes
    ----------
    map_size : int
        The size of the map.
    fov_pad : int
        The amount of padding applied to each agent's FOV.
    number_of_agents : int
        The number of agents.
    number_of_obstacles : int
        The number of obstacles.
    difficulty : float
        The difficulty in (0, 1).
    duration : int
        The duration of episodes.

    Example
    -------
    >>> configuration = AC2Configuration(
    ...     map_size=16,
    ...     fov_pad=2,
    ...     number_of_agents=16,
    ...     number_of_obstacles=16,
    ...     difficulty=0.01,
    ...     duration=1000,
    ... )
    """

    map_size: int
    fov_pad: int
    number_of_agents: int
    number_of_obstacles: int
    difficulty: float
    duration: int


gem = gem_policy(None)

class AC2(AECEnv):
    """AC2.

    This is an AEC environment Ant Colony Coverage (AC2). In AC2, the goal is
    for a colony of ants to distribute pheromone uniformly over a surface. The
    reward at each step is the perplexity of the pheromone distribution.

    Attributes
    ----------
    configuration : AC2Configuration
        The AC2 configuration.

    Example
    -------
    >>> configuration = AC2Configuration(
    ...     map_size=16,
    ...     fov_pad=2,
    ...     number_of_agents=16,
    ...     number_of_obstacles=16,
    ...     difficulty=0.01,
    ...     duration=1000,
    ... )
    >>> env = AC2(configuration=configuration)
    """

    def __init__(self, configuration: AC2Configuration) -> None:
        """Initialize the environment.

        Parameters
        ----------
        configuration : AC2Configuration
            The AC2 configuration.
        """

        super().__init__()

        self.configuration = configuration

        # AEC attributes.

        self.metadata = {'is_parallelizable': True}

        agents = list(range(self.configuration.number_of_agents))
        agents = [str(agent) for agent in agents]

        self.agents = agents
        self.possible_agents = agents
        self.render_mode = None

    # AEC methods.

    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        """Return the observation space for an agent."""

        fov_size = (self.configuration.fov_pad * 2) + 1

        return gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(3, fov_size, fov_size),
        )

    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        """Return the action space for an agent."""

        return gym.spaces.Discrete(4)

    def step(self, action: ActionType) -> None:
        """Step an agent."""

        if action == None:
            self.terminations = {agent: False for agent in self.agents}
            self.truncations = self.terminations

            return

        # self.rewards[self.agent_selection] = 1. if action == self._gem_actions[self.agent_selection] else -1.

        x, y = self._agent_coordinates[self._agent_selection]
        x1, y1 = x, y

        if action == ACTION_NORTH: x1, y1 = x, y - 1
        elif action == ACTION_EAST: x1, y1 = x + 1, y
        elif action == ACTION_SOUTH: x1, y1 = x, y + 1
        elif action == ACTION_WEST: x1, y1 = x - 1, y

        # Move agent if valid.

        is_valid_action = (y1 in range(0, self.configuration.map_size)) \
            and (x1 in range(0, self.configuration.map_size)) \
            and (self._map[OBSTACLE_CHANNEL, y1, x1] == 0.) \
            and (self._map[AGENT_CHANNEL, y1, x1] == 0.) \

        if is_valid_action:
            self._map[AGENT_CHANNEL, y, x] = 0.
            self._map[AGENT_CHANNEL, y1, x1] = 1.
            self._agent_coordinates[self._agent_selection] = (x1, y1)

        # If this was the last agent, then update the pheromone and reward. We
        # only update after having moved every agent in order to make the
        # environment parallelizable.

        if self._agent_selection + 1 == self.configuration.number_of_agents:
            self._map[PHEROMONE_CHANNEL] *= 1 - self.configuration.difficulty
            self._map[PHEROMONE_CHANNEL] += self._map[AGENT_CHANNEL]

            ppl = PPL(self._map[PHEROMONE_CHANNEL]) / self.configuration.number_of_agents
            reward = (ppl - self._ppl) #* 1/(self._step)
            self._ppl = ppl

            self._clear_rewards()
            self.rewards = {agent: reward for agent in self.agents}
            self._accumulate_rewards()

            self._step += 1

            if self._step == self.configuration.duration:
                self.terminations = {agent: True for agent in self.agents}
                self.truncations = self.terminations

            if self._step > self.configuration.duration:
                raise AssertionError(
                    'Step exceeds duration. In AC2, you must call `reset()` once'
                    ' the first agent has `termination=True`.'
                )

        # Increment agent selection.

        self._agent_selection = (self._agent_selection + 1) \
            % self.configuration.number_of_agents

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional = None,
    ) -> None:
        """Reset the environment."""

        ## Initialize the map.

        coordinates = {}
        number_of_coordinates = self.configuration.number_of_agents \
            + self.configuration.number_of_obstacles

        while len(coordinates) < number_of_coordinates:
            coordinate = (
                random.randint(0, self.configuration.map_size - 1),
                random.randint(0, self.configuration.map_size - 1),
            )

            if coordinate not in coordinates:
                coordinates[coordinate] = True

        coordinates = list(coordinates)

        self._agent_coordinates, self._obstacle_coordinates = (
            coordinates[: self.configuration.number_of_agents],
            coordinates[self.configuration.number_of_agents :],
        )

        self._map = np.zeros((
            3,
            self.configuration.map_size,
            self.configuration.map_size,
        ))

        self._map[
            OBSTACLE_CHANNEL,
            np.array(self._obstacle_coordinates)[:, 1],
            np.array(self._obstacle_coordinates)[:, 0],
        ] = 1.

        self._map[
            AGENT_CHANNEL,
            np.array(self._agent_coordinates)[:, 1],
            np.array(self._agent_coordinates)[:, 0],
        ] = 1.

        self._map[PHEROMONE_CHANNEL] = self._map[AGENT_CHANNEL].copy()

        # Initialize AEC attributes.

        agents = self.agents

        self._agent_selection = 0
        self._cumulative_rewards = {agent: 0. for agent in agents}
        self.rewards = {agent: 0. for agent in agents}
        self.terminations = {agent: False for agent in agents}
        self.truncations = self.terminations

        # Initialize remaining attributes.

        self._step = 0
        self._ppl = 0.
        self._gem_actions = {}

    def observe(self, agent: AgentID) -> ObsType:
        """Return the observation for the selected agent."""

        # Reconstruct map with padding.

        agents = np.pad(
            array=self._map[AGENT_CHANNEL],
            pad_width=self.configuration.fov_pad,
            mode='constant',
            constant_values=0.,
        )

        obstacles = np.pad(
            array=self._map[OBSTACLE_CHANNEL],
            pad_width=self.configuration.fov_pad,
            mode='constant',
            constant_values=1.,
        )

        pheromone = np.pad(
            array=self._map[PHEROMONE_CHANNEL],
            pad_width=self.configuration.fov_pad,
            mode='constant',
            constant_values=0.,
        )

        map = np.zeros((3, *agents.shape))
        map[AGENT_CHANNEL] = agents
        map[OBSTACLE_CHANNEL] = obstacles
        map[PHEROMONE_CHANNEL] = normalize(pheromone)

        # Create a local view for the current agent.

        x, y = self._agent_coordinates[int(agent)]
        x = x + self.configuration.fov_pad
        y = y + self.configuration.fov_pad

        fov = map[
            :,
            y - self.configuration.fov_pad : y + self.configuration.fov_pad + 1,
            x - self.configuration.fov_pad : x + self.configuration.fov_pad + 1,
        ]

        # self._gem_actions[agent] = gem(fov)

        return fov

    def state(self) -> np.ndarray:
        return self._map.copy()

    @property
    def infos(self) -> Dict[AgentID, Dict[str, Any]]:
        return {
            str(agent): {
                'id': agent,
                'coordinate': self._agent_coordinates[int(agent)],
            } for agent in self.agents
        }

    @property
    def agent_selection(self) -> AgentID:
        return str(self._agent_selection)
