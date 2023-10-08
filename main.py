from chatarena.agent import Player, Moderator
from chatarena.backends import OpenAIChat, Claude, CohereAIChat, TransformersConversational
from chatarena.message import Message, MessagePool
from chatarena.environments import Environment, Chameleon, Conversation, PettingzooChess, PettingzooTicTacToe
from chatarena.arena import Arena
from chatarena.environments.avalon import Avalon

import argparse
import os
api_key = os.environ.get('OPENAI_API_KEY')
# anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')

from collections import defaultdict
import random
random.seed(42)

def load_prompts(path):
    prompts = defaultdict(str)
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        
        if os.path.isfile(filepath):
            prompt = open(filepath, "r").read()
            prompts[filename.split(".txt")[0]] = prompt
    print(len(prompts))
    print(prompts.keys())
    return prompts

def run_main():

    NUM_PLAYERS = 5 # Merlin, Percival, Morgana, Assassin, Servant
    PLAYER_NAMES = ["Merlin", "Percival", "Morgana", "Assassin", "Servant"]
    NUM_QUESTS = 5
    SIZE_OF_QUESTS = [2, 3, 2, 3, 3]

    prompt_path = "chatarena/prompt_templates"
    prompts = load_prompts(prompt_path)

    # create players
    names = random.sample(PLAYER_NAMES, len(PLAYER_NAMES))
    print(names)
    players = []
    for i in range(len(names)):
        print(names[i])
        role_i = prompts["environment_description"] + "\n\n" + \
                    prompts["role_description"].format(player_i=i, role_i=names[i], num_other_players=NUM_PLAYERS-1) # + \
                    # prompts["format_specification"]
        print(role_i)
        player_i = Player(name=names[i],
                        role_desc=role_i,
                        #backend=OpenAIChat(model="gpt-3.5-turbo"))
                        backend=TransformersConversational(model="facebook/blenderbot-400M-distill"))
        players.append(player_i)
    print(players)
    
    role_moderator = prompts["moderator_description"].format(num_other_players=NUM_PLAYERS)
    print(role_moderator)
    moderator = Moderator(
                        role_desc=role_moderator,
                        terminal_condition=prompts["terminal_condition"],
                        #backend=OpenAIChat(model="gpt-3.5-turbo"))
                        backend=TransformersConversational(model="facebook/blenderbot-400M-distill"))
    print(moderator)

    

if __name__ == "__main__":
    run_main()