from chatarena.agent import Player, Moderator
from chatarena.backends import OpenAIChat, Claude, CohereAIChat, TransformersConversational
from chatarena.message import Message, MessagePool
# from chatarena.environments import Environment, Avalon
from chatarena.arena import Arena
from chatarena.environments.avalon import Avalon

import argparse
import os
import openai


api_key = os.environ.get('OPENAI_API_KEY')
print(api_key)
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
    #print(prompts.keys())
    return prompts

def run_main():

    PLAYER_NAMES = ["Merlin", "Percival", "Morgana", "Assassin", "Servant"]
    SIZE_OF_QUESTS = [2, 3, 2, 3, 3]

    prompt_path = "chatarena/prompt_templates"
    prompts = load_prompts(prompt_path)

    # create players
    names = random.sample(PLAYER_NAMES, len(PLAYER_NAMES))
    anonymous_names = [f"Player {i}" for i in range(len(PLAYER_NAMES))]
    #print(anonymous_names)
    print(f"{anonymous_name}: {name}" for anonymous_name, name in zip(anonymous_names, names))

    players = []
    for i in range(len(names)):
        #print(names[i])
        role_i = prompts["environment_description"] + "\n\n" + \
                    prompts["role_description"].format(player_i=i, role_i=names[i], num_other_players=len(PLAYER_NAMES)-1) # + \
                    # prompts["format_specification"]
        #print(role_i)
        player_i = Player(name=anonymous_names[i],
                          role=names[i],
                        role_desc=role_i,
                        backend=OpenAIChat(model="gpt-3.5-turbo"))
                        #backend=TransformersConversational(model="facebook/blenderbot-400M-distill"))
        players.append(player_i)
    print(players)
    
    role_moderator = prompts["moderator_description"].format(num_other_players=len(PLAYER_NAMES))
    print(role_moderator)
    moderator = Moderator(role="Moderator",
                        role_desc=role_moderator,
                        terminal_condition=prompts["terminal_condition"],
                        backend=OpenAIChat(model="gpt-3.5-turbo"))
                        #backend=TransformersConversational(model="facebook/blenderbot-400M-distill"))
    print(moderator)

    env = Avalon(names, SIZE_OF_QUESTS)
    current_leader = env.current_leader_idx
    print(current_leader, names[env.current_leader_idx])

    MAX_STEPS = 100
    arena = Arena(players, env)
    arena.launch_cli(max_steps=MAX_STEPS, interactive=False)


if __name__ == "__main__":
    run_main()