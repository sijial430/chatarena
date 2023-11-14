from chatarena.agent import Player, Moderator
from chatarena.backends import OpenAIChat, Claude, CohereAIChat, TransformersConversational
from chatarena.message import Message, MessagePool
# from chatarena.environments import Environment, Avalon
from chatarena.arena import Arena
from chatarena.environments.avalon import Avalon

import argparse
import os
import openai
import json
from nltk.tokenize import word_tokenize, sent_tokenize

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
            if filename.endswith(".txt"):
                prompts[filename.split(".txt")[0]] = prompt
            elif filename.endswith(".json"):
                prompts[filename.split(".json")[0]] = json.loads(prompt)
    #print(prompts.keys())
    return prompts

def run_main():

    PLAYER_NAMES = ["Merlin", "Percival", "Morgana", "Assassin", "Loyal Servant of Arthur"]
    SIZE_OF_QUESTS = [2, 3, 2, 3, 3]

    #prompt_path = "chatarena/prompt_templates"
    prompt_path = "chatarena/prompt_templates/recon"
    prompts = load_prompts(prompt_path)

    # create players
    #names = random.sample(PLAYER_NAMES, len(PLAYER_NAMES))

    names = ["Merlin", "Percival", "Loyal Servant of Arthur", "Assassin", "Morgana"]

    anonymous_names = [f"Player {i}" for i in range(len(PLAYER_NAMES))]
    #print(anonymous_names)
    #print(f"{anonymous_name}: {name}" for anonymous_name, name in zip(anonymous_names, names))

    players = []
    avg_prompt_len = []
    current_situation = "Starting game"
    for i in range(len(names)):
        #print(names[i])
        '''
        role_prompt = prompts['environment_description'].format(player_i=i, num_players=len(names))
        '''
        role_prompt = prompts['game_rule_prompt'] \
            + "\n\n" \
            + prompts['first_order_prompt'].format(player_i=i, role_i=names[i], current_situation=current_situation) \
            + "\n\nHints for your role:\n" \
            + prompts['role_hints_prompt'][names[i]] #\
            #+ "\n\n" \
            #+ prompts['formulation_prompt']
        #'''
        avg_prompt_len.append(len(word_tokenize(role_prompt)))
        player_i = Player(name=anonymous_names[i],
                          role=names[i],
                          role_desc=role_prompt,
                          backend=OpenAIChat(model="gpt-3.5-turbo-1106"))
        players.append(player_i)
    print(players)
    print(sum(avg_prompt_len) / len(avg_prompt_len))
    
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

    MAX_STEPS = 200
    arena = Arena(players, env)
    arena.launch_cli(max_steps=MAX_STEPS, interactive=False)


if __name__ == "__main__":
    run_main()
