from chatarena.environments.base import Environment, TimeStep
#from chatarena.environments.conversation import Conversation, ModeratedConversation
from chatarena.environments.chameleon import Chameleon
#from chatarena.environments.pettingzoo_chess import PettingzooChess
#from chatarena.environments.pettingzoo_tictactoe import PettingzooTicTacToe
from chatarena.environments.avalon import Avalon

from chatarena.config import EnvironmentConfig

ALL_ENVIRONMENTS = [
    #Conversation,
    #ModeratedConversation,
    Chameleon,
    #PettingzooChess,
    #PettingzooTicTacToe,
    Avalon
]

ENV_REGISTRY = {env.type_name: env for env in ALL_ENVIRONMENTS}


# Load an environment from a config dictionary
def load_environment(config: EnvironmentConfig):
    try:
        env_cls = ENV_REGISTRY[config["env_type"]]
    except KeyError:
        raise ValueError(f"Unknown environment type: {config['env_type']}")

    env = env_cls.from_config(config)
    return env
