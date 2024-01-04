from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import math
import random

import numpy as np

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


# PPL reward.

def PPL(x: np.ndarray) -> float:
    """Compute the perplexity."""

    distribution = (x + 1e-8) / (x + 1e-8).sum()
    entropy = (np.log(distribution) * distribution).sum()
    perplexity = np.exp(entropy)

    return perplexity


# Environment configuration.

@dataclass(frozen=True)
class AC2Configuration:
    """AC2 configuration.

    Attributes
    ----------
    map_size : int
        The size of the map.
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
    ...     number_of_agents=16,
    ...     number_of_obstacles=16,
    ...     difficulty=0.01,
    ...     duration=1000,
    ... )
    """

    map_size: int
    number_of_agents: int
    number_of_obstacles: int
    difficulty: float
    duration: int


# Environment.

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

    # AEC methods.

    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        """Return the observation space for an agent."""

        return gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(3, self.configuration.map_size, self.configuration.map_size),
        )

    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        """Return the action space for an agent."""

        return gym.spaces.Discrete(4)

    def step(self, action: ActionType) -> None:
        """Step an agent."""

        x, y = self._agent_coordinates[self.agent_selection]
        x1, y1 = x, y

        if action == ACTION_NORTH: x1, y1 = x, y - 1
        elif action == ACTION_EAST: x1, y1 = x + 1, y
        elif action == ACTION_SOUTH: x1, y1 = x, y + 1
        elif action == ACTION_WEST: x1, y1 = x - 1, y

        # Move agent if valid.

        is_valid_action = (y in range(0, self.configuration.map_size)) \
            and (x in range(0, self.configuration.map_size)) \
            and (self._map[OBSTACLE_CHANNEL, y1, x1] == 0.) \
            and (self._map[AGENT_CHANNEL, y1, x1] == 0.) \
        
        if is_valid_action:
            self._map[AGENT_CHANNEL, y, x] = 0.
            self._map[AGENT_CHANNEL, y1, x1] = 1.
            self._agent_coordinates[self.agent_selection] = (x1, y1)

        # If this was the last agent, then update the pheromone and reward. We
        # only update after having moved every agent in order to make the 
        # environment parallelizable.

        if self.agent_selection + 1 == self.configuration.number_of_agents:
            self._map[PHEROMONE_CHANNEL] *= 1 - self.configuration.difficulty
            self._map[PHEROMONE_CHANNEL] += self._map[AGENT_CHANNEL]

            ppl = PPL(self._map[PHEROMONE_CHANNEL]) / self.configuration.number_of_agents

            self._clear_rewards()
            self.rewards = {agent: ppl for agent in self.agents}
            self._accumulate_rewards()

            self._step += 1

            if self._step == self.configuration.duration:
                self.terminations = {agent: True for agent in self.agents}
                self.truncations = self.terminations

            if self._step > self.configuration.duration:
                raise AssertionError(
                    'Step exceeds duration. In AC2, you must call `reset()` once'
                    'the first agent has `termination=True`.'
                )

        # Increment agent selection.

        self.agent_selection = (self.agent_selection + 1) \
            % self.configuration.number_of_agents

    def reset(self, seed: Optional[int] = None) -> None:
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

        agents = list(range(self.configuration.number_of_agents))

        self.agent_selection = 0
        self._cumulative_rewards = {agent: 0. for agent in agents}

        self.agents = agents
        self.possible_agents = agents
        self.rewards = {agent: 0. for agent in agents}
        self.terminations = {agent: False for agent in agents}
        self.truncations = self.terminations

        # Initialize remaining attributes.

        self._step = 0

    def observe(self, agent: AgentID) -> ObsType:
        """Return the observation for the selected agent."""

        return self._map
    
    def state(self) -> np.ndarray:
        return self._map
    
    @property
    def infos(self) -> Dict[AgentID, Dict[str, Any]]:
        return {
            agent: {
                'id': agent,
                'coordinate': self._agent_coordinates[agent],
            } for agent in self.agents
        }
