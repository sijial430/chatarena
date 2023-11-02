from typing import List, Union
import re
from tenacity import RetryError
import logging
import uuid
from abc import abstractmethod
import asyncio

from .backends import IntelligenceBackend, load_backend
from .message import Message, SYSTEM_NAME
from .config import AgentConfig, Configurable, BackendConfig

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):
    """
        An abstract base class for all the agents in the chatArena environment.
    """
    @abstractmethod
    def __init__(self, name: str, role: str, role_desc: str, global_prompt: str = None, *args, **kwargs):
        """
        Initialize the agent.

        Parameters:
            name (str): The name of the agent.
            role_desc (str): Description of the agent's role.
            global_prompt (str): A universal prompt that applies to all agents. Defaults to None.
        """
        super().__init__(name=name, role=role, role_desc=role_desc, global_prompt=global_prompt, **kwargs)
        self.name = name
        self.role = role
        self.role_desc = role_desc
        self.global_prompt = global_prompt

class Player(Agent):
    """
    The Player class represents a player in the chatArena environment. A player can observe the environment
    and perform an action (generate a response) based on the observation.
    """

    def __init__(self, name: str, role: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, **kwargs):
        """
        Initialize the player with a name, role description, backend, and a global prompt.

        Parameters:
            name (str): The name of the player.
            role_desc (str): Description of the player's role.
            backend (Union[BackendConfig, IntelligenceBackend]): The backend that will be used for decision making. It can be either a LLM backend or a Human backend.
            global_prompt (str): A universal prompt that applies to all players. Defaults to None.
        """

        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        super().__init__(name=name, role=role, role_desc=role_desc, backend=backend_config,
                         global_prompt=global_prompt, **kwargs)

        self.backend = backend

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role=self.role,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )

    def act(self, observation: List[Message]) -> str:
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dyanmics.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        return self.act_multistep(observation)

        try:
            request_message = f"Now you speak, {self.name}. Carefully consider your role, the roles of the other players, your objectives for this round and for the entire game, and the opinions of the other players. Craft your response to maximize the changes of convincing the other players to go with your plan, to ensure your goals are achieved and your team wins the game.<EOS>"
            response = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=request_message)
        except RetryError as e:
            err_msg = f"Agent {self.role} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response

    def act_multistep(self, observation):
        try:
            thought_msg1 = f"What is the state of the game? Make sure your answer touches on the other players' roles, thoughts and plans."
            thought_msg2 = f"What is your objective in this round? Consider your role, the game state, and your chances of winning the game."
            thought_msg3 = f"How can you achieve your objective for this round?"
            thought_msg4 = f"Craft a message to send to the other players. Ensure your responses helps move forward the game according to your objectives. Ensure not to reveal too much in your message."
            thought_msg5 = f"Now consider your strategy and objective, and revise your message. If your message is too long or says anything about your identity, ensure to fix it."

            thought_msg_list1 = [thought_msg1]
            response1 = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, thought_msgs=thought_msg_list1)

            thought_msg_list2 = [thought_msg1, response1, thought_msg2]
            response2 = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, thought_msgs=thought_msg_list2)

            thought_msg_list3 = [thought_msg1, response1, thought_msg2, response2, thought_msg3]
            response3 = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, thought_msgs=thought_msg_list3)

            thought_msg_list4 = [thought_msg1, response1, thought_msg2, response2, thought_msg3, response3, thought_msg4]
            response4 = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, thought_msgs=thought_msg_list4)

            thought_msg_list5 = [thought_msg1, response1, thought_msg2, response2, thought_msg3, response3, thought_msg4, response4, thought_msg5]
            response5 = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, thought_msgs=thought_msg_list5)
            print('\n'.join(thought_msg_list5))
            print(response5)
            response = response5
        except RetryError as e:
            err_msg = f"Agent {self.role} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response

    def __call__(self, observation: List[Message]) -> str:
        return self.act(observation)

    async def async_act(self, observation: List[Message]) -> str:
        """
        Async version of act(). This is used when you want to generate a response asynchronously.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        try:
            response = self.backend.async_query(agent_name=self.name, role=self.role, role_desc=self.role_desc,
                                                history_messages=observation, global_prompt=self.global_prompt,
                                                request_msg=None)
        except RetryError as e:
            err_msg = f"Agent {self.role} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response

    def reset(self):
        """
        Reset the player's backend in case they are not stateless.
        This is usually called at the end of each episode.
        """
        self.backend.reset()


class Moderator(Player):
    """
    The Moderator class represents a special type of player that moderates the conversation.
    It is usually used as a component of the environment when the transition dynamics is conditioned on natural language that are not easy to parse programatically.
    """

    def __init__(self, role: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 terminal_condition: str, global_prompt: str = None, **kwargs):
        """
        Initialize the moderator with a role description, backend, terminal condition, and a global prompt.

        Parameters:
            role_desc (str): Description of the moderator's role.
            backend (Union[BackendConfig, IntelligenceBackend]): The backend that will be used for decision making.
            terminal_condition (str): The condition that signifies the end of the conversation.
            global_prompt (str): A universal prompt that applies to the moderator. Defaults to None.
       """
        name = "Moderator"
        super().__init__(name=name, role=role, role_desc=role_desc, backend=backend, global_prompt=global_prompt, **kwargs)

        self.terminal_condition = terminal_condition

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role=self.role,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            terminal_condition=self.terminal_condition,
            global_prompt=self.global_prompt,
        )

    def is_terminal(self, history: List[Message], *args, **kwargs) -> bool:
        """
        Check whether an episode is terminated based on the terminal condition.

        Parameters:
            history (List[Message]): The conversation history.

        Returns:
            bool: True if the conversation is over, otherwise False.
        """
        # If the last message is the signal, then the conversation is over
        if history[-1].content == SIGNAL_END_OF_CONVERSATION:
            return True

        try:
            request_msg = Message(agent_name=self.name, content=self.terminal_condition, turn=-1)
            response = self.backend.query(agent_name=self.name, role=self.role, role_desc=self.role_desc, history_messages=history,
                                          global_prompt=self.global_prompt, request_msg=request_msg, *args, **kwargs)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}.")
            return True

        if re.match(r"yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", response, re.IGNORECASE):
            # print(f"Decision: {response}. Conversation is ended by moderator.")
            return True
        else:
            return False
