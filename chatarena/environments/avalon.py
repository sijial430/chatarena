from chatarena.environments.base import TimeStep, Environment
from chatarena.message import Message, MessagePool
from chatarena.utils import extract_jsons
from typing import List, Union

from typing import List, Dict
from abc import abstractmethod

from ..message import Message
from ..config import EnvironmentConfig


class Avalon(Environment):
    type_name = "avalon"

    def __init__(self, max_quests:int):
        super().__init__(player_names=["Merlin", "Percival", "Morgana", "Assassin", "Servant"])
        self.max_quests = max_quests
        self.quest = 0
        self.success_quests = 0
        self.failed_quests = 0
        self.failed_consecutive_votes = 0
        # TODO: ADD PLAYER INFO
        self.assasinated_player = None
        self.message_pool = MessagePool()
        self._terminal = False
        self.reset()

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """
        moderator say something to inform the game status and collect quest results
        """
        message = Message(agent_name="Moderator", content=text, turn=self.turn, visible_to=visible_to)
        self.message_pool.append_message(message)
    
    @abstractmethod
    def reset(self):
        self.quest = 0
        self.success_quests = 0
        self.failed_quests = 0
        self.failed_consecutive_votes = 0
        self.assasinated_player = None
        self.message_pool.reset()
        self._terminal = False
        # Moderator declares the start of game
        self._moderator_speak(f"All players close your eyes")
        self._moderator_speak(f"Minions of Mordred, these players are all agents of Evil.", visible_to=["Morgana","Assassin"])
        self._moderator_speak(f"Merlin, these players are the agents of Evil.", visible_to="Merlin")
        self._moderator_speak(f"Percival, these two players are either Merlin or Morgana", visible_to="Percival")
        self._moderator_speak(f"All players open your eyes")
        observation = self.get_observation(self.get_next_player())
        return TimeStep(observation=observation, reward=self._get_zero_rewards(), terminal=False)

    @abstractmethod
    def get_observation(self, player_name=None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(player_name, turn=self.turn + 1)

    def to_config(self) -> EnvironmentConfig:
        self._config_dict["env_type"] = self.type_name
        return EnvironmentConfig(**self._config_dict)

    @property
    def num_players(self) -> int:
        """
        get the number of players
        """
        return len(self.player_names)

    @abstractmethod
    def get_next_player(self) -> str:
        """
        Return the name of the next player.

        Note:
            This method must be implemented by subclasses.

        Returns:
            str: The name of the next player.
        """
        pass

    @abstractmethod
    def get_observation(self, player_name=None) -> List[Message]:
        """
        Return observation for a given player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            player_name (str, optional): The name of the player for whom to get the observation.

        Returns:
            List[Message]: The observation for the player in the form of a list of messages.
        """
        pass

    @abstractmethod
    def print(self):
        """
        print the environment state
        """
        pass

    @abstractmethod
    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Execute a step in the environment given an action from a player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            player_name (str): The name of the player.
            action (str): The action that the player wants to take.

        Returns:
            TimeStep: An object of the TimeStep class containing the observation, reward, and done state.
        """
        pass

    @abstractmethod
    def check_action(self, action: str, player_name: str) -> bool:
        """
        Check whether a given action is valid for a player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            action (str): The action to be checked.
            player_name (str): The name of the player.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        return True

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check whether the environment is in a terminal state (end of episode).

        Note:
            This method must be implemented by subclasses.

        Returns:
            bool: True if the environment is in a terminal state, False otherwise.
        """
        pass

    def get_zero_rewards(self) -> Dict[str, float]:
        """
        Return a dictionary with all player names as keys and zero as reward.

        Returns:
            Dict[str, float]: A dictionary of players and their rewards (all zero).
        """
        return {player_name: 0. for player_name in self.player_names}

    def get_one_rewards(self) -> Dict[str, float]:
        """
        Return a dictionary with all player names as keys and one as reward.

        Returns:
            Dict[str, float]: A dictionary of players and their rewards (all one).
        """
        return {player_name: 1. for player_name in self.player_names}
