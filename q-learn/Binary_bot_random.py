# -*- coding: utf-8 -*-
"""
@author: Eric Born
Developed a bot that plays at the Protoss race
Choses a random difficulty between 0-9 then launches SC2
and plays against the built-in AI
Keeps track 12 attributes of the games progress and writes the results
out to a numpy array file
Also appends the outcome of the match to a csv file. 
-1 for loss, 0 for tie, 1 for win
"""
# general libraries
import numpy as np
import pandas as pd
import random
import keras
import math
import time
import csv
import os

# import asyncio
# from absl import app

# pysc2 libraries
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

# sc2 libraries
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, \
 CYBERNETICSCORE, GATEWAY, ROBOTICSBAY, ROBOTICSFACILITY, STARGATE, \
 ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, COLOSSUS

# Q learning system found here:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01,
#                  reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon = e_greedy
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

#     def choose_action(self, observation):
#         self.check_state_exist(observation)

#         if np.random.uniform() < self.epsilon:
#             # choose best action
#             state_action = self.q_table.ix[observation, :]

#             # some actions have the same value
#             state_action = state_action.reindex(np.random.permutation(
#                                                 state_action.index))

#             action = state_action.idxmax()
#         else:
#             # choose random action
#             action = np.random.choice(self.actions)

#         return action

#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         self.check_state_exist(s)

#         q_predict = self.q_table.ix[s, a]
#         q_target = r + self.gamma * self.q_table.ix[s_, :].max()

#         # update
#         self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # append new state to q table
#             self.q_table = self.q_table.append(pd.Series([0] *
#                                                len(self.actions),
#                                                index=self.q_table.columns,
#                                                name=state))

#choice = random.randint(0, 8)

# TODO
#### USE DESCRETE ACTIONS AS INPUTS TO THE RL algorithm 
# Dueling-DDQN [7, 38, 39] and PPO [40]), together with a distributed rollout infrastructure.


# TODO
#### If an action is chosen but cannot be completed due to tech being locked
#### Lack of resources, etc. The do_nothing action should be taken instead
#### Possibly have an expanding tree that when certain criteria are met, the actions can then be selected???

# TODO
# AT EACH STEP, AGENT RECEIVES OBSERVATIONS THEN CHOOSES AN MACRO ACTION
# ONCE ACTION IS CHOSEN, IT FILTERS DOWN TO THE FUNCTION THAT HANDLES THE MICRO ACTIONS.
# EG. ACTION_OFFENSIVE_FORCE_BUILDINGS, FUNCTION CHECKS IF PYLON EXISTS, IF NOT IT BUILDS PYLON. 
# IF YES THEN DOES GATEWAY EXIST, IF YES DOES CYBER CORE EXIST, ETC.

# TODO
# TENCENT IS USING A REWARD SYSTEM AFTER EACH STEP, SEE IF YOU CAN FIND WHAT IT IS.
# INFORMATION SEEMS CONFLICTING, ONE SAYS AFTER EACH STEP THEN REFER TO A SECTION THAT
# ONLY COVERS THE END OF GAME REWARD.
# POSSIBLY CREATE A SYSTEM THAT PENALIZES CHOOSING AN OPTION THAT ISNT CURRENTLY AVAILABLE
# SO THAT THE BOT IS LESS LIKELY TO CHOSE IT ON THE NEXT STEP. AFTER IT COMPLETES A STEP
# THE PENALTY GOES AWAY.

# Creates variables holding strings for bots actions
# ACTION_ATTACK = 'attack'
# ACTION_BUILD_ASSIMILATORS = 'build_assimilators'
# ACTION_BUILD_OFFENSIVE_FORCE = 'build_offensive_force'
# ACTION_BUILD_PYLONS = 'build_pylons'
# ACTION_BUILD_WORKERS = 'build_workers'
# ACTION_DISTRIBUTE_WORKERS = 'distribute_workers'
# ACTION_DO_NOTHING = 'nothing'
# ACTION_EXPAND = 'expand'
# ACTION_OFFENSIVE_FORCE_BUILDINGS = 'offensive_force_buildings'

# creates a list of actions the bot can choose from
# smart_actions = [
#     ACTION_ATTACK,
#     ACTION_BUILD_ASSIMILATORS,
#     ACTION_BUILD_OFFENSIVE_FORCE,
#     ACTION_BUILD_PYLONS,
#     ACTION_BUILD_WORKERS,
#     ACTION_DISTRIBUTE_WORKERS,
#     ACTION_DO_NOTHING,
#     ACTION_EXPAND,
#     ACTION_OFFENSIVE_FORCE_BUILDINGS
# ]



# units unlocked by the following buildings
# gateway - ZEALOT
# cyber core - STALKER, ADEPT
# robo facility - IMMORTAL
# stargate - VOIDRAY
# robo bay - COLOSSUS
unit_list = [
    None,
    'ZEALOT',
    'STALKER',
    'ADEPT',
    'IMMORTAL',
    'VOIDRAY',
    'COLOSSUS',
]

# building indicators, used to check if units can be created
# Flipped to a 1 if they exist
GATEWAY_IND = 0
CYBERCORE_IND = 0
ROBOFACILITY_IND = 0
STARGATE_IND = 0
ROBOBAY_IND = 0

unit_choice = ''

# possible rewards or counters for the learning system
# units_destroyed = 0

# Define rewards for killing units or buildings
# KILL_UNIT_REWARD = 0.2
# KILL_BUILDING_REWARD = 0.5
# SUPPLY_ARMY_REWARD = 0.2
# SUPPLY_WORKERS_REWARD = 0.05

# creates empty np array to store game stats, count of 
# actions taken, match difficulty and the overall game outcome
# [0]supply_cap, [1]supply_army, [2]supply_workers, [3]PYLON, [4]ASSIMILATOR, 
# [5]GATEWAY, [6]NEXUS, [7]killed_structures, [8]killed_units, 
# actions
# [9]attack, [10]assimilators, [11]offensive_force, [12]nothing, [13]workers, 
# [14]pylons, [15]nothing, [16]expand, [17]buildings,
# [18]difficulty, [19]outcome


# Creates a random number between 0-9
# this is used in the main() to set the difficulty of the game
diff = random.randrange(0,10)




# Isnt working because you cant pass the difficulty as a string
# diff_list = [
#     'Difficulty.VeryEasy', 'Difficulty.Easy', 'Difficulty.Medium,'
#     'Difficulty.MediumHard', 'Difficulty.Hard', 'Difficulty.Harder',
#     'Difficulty.VeryHard', 'Difficulty.CheatMoney', 'Difficulty.CheatVision', 
#     'Difficulty.CheatInsane'
# ]

# maps the functions from the pysc2 actions file
FUNCTIONS = actions.FUNCTIONS

# Bots current issues:
# Troops need to move out to protect expanded bases
# sometimes creates multiple nexuses right next to each other
# wont target troops repairing buildings
class BinaryBot(sc2.BotAI):
    def __init__(self, use_model=False):
        #self.ITERATIONS_PER_MINUTE = 165 
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.delay = 1
        self.use_model = use_model
        self.score_data = np.zeros(20)

        # Store the difficulty setting in the array that is used as output data
        self.score_data[18] = diff

        # Setup actions dictionary
        self.actions = {
            0: self.attack,
            1: self.build_assimilators,
            2: self.build_offensive_force,
            3: self.build_pylons,
            4: self.build_workers,
            5: self.distribute_workers,
            6: self.do_nothing,
            7: self.expand,
            8: self.offensive_force_buildings
        }
        self.actions_data = []

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("model-name")
        
        # self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

    # Create a function to write the result to a csv
    def write_csv(self, game_result):
        with open('record.csv','a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(str(game_result))

    def on_end(self, game_result):
        result = str(game_result)
        
        # Defeat
        if result == 'Result.Defeat':
            self.score_data[19] = -1
            self.write_csv(str(-1))
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.score_data))
        
        # Win
        elif result == 'Result.Victory':
            self.score_data[19] = 1
            self.write_csv(1)
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.score_data))
        
        # Tie
        else:
            self.score_data[19] = 0
            self.write_csv(0)
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.score_data))

    # This is the function steps forward and is called through each frame of the game
    async def on_step(self, iteration):
        #self.iteration = iteration
        self.time_loop = (self.state.game_loop/22.4) / 60
        #print(self.time_loop)
        await self.smart_action()
        await self.back_to_work()
        await self.do_nothing()
        # if self.time > self.do_something_after:
        #     choice = smart_actions[random.randint(0, 8)]
        #     print('my choice is', choice)
        # random number selected on each step that dictates the bots action
        
        # send starting chat message
        if iteration == 0:
            await self.chat_send("(glhf)")

        # TODO 
        # May need to record this data every so many steps throughout the game
        # instead of just once at the end
        
        # checks to see if the value has increased, if so records new value
        # supplies
        if iteration % 5 == 0:
            self.score_data[0] = self.supply_cap
            self.score_data[1] = self.supply_army
            self.score_data[2] = self.supply_workers
            self.score_data[3] = self.units(PYLON).amount
            self.score_data[4] = self.units(ASSIMILATOR).amount
            self.score_data[5] = self.units(GATEWAY).amount
            self.score_data[6] = self.units(NEXUS).amount
            self.score_data[7] = self.state.score.killed_value_structures
            self.score_data[8] = self.state.score.killed_value_units

        # if choice == 'attack':
        #     #self.attack()
        #     self.score_data[9] += 1

        # elif choice == 'build_assimilators':
        #     #self.build_assimilators
        #     self.score_data[10] += 1

        # elif choice == 'build_offensive_force':
        #     #self.build_offensive_force
        #     self.score_data[11] += 1

        # elif choice == 'build_pylons':
        #     #self.build_pylons
        #     self.score_data[12] += 1

        # elif choice == 'build_workers':
        #     #self.build_workers
        #     self.score_data[13] += 1

        # elif choice == 'distribute_workers':
        #     #self.distribute_workers
        #     self.score_data[14] += 1

        # elif choice == 'nothing':
        #     #self.do_nothing
        #     self.score_data[15] += 1

        # elif choice == 'expand':
        #     #self.expand
        #     self.score_data[16] += 1

        # elif choice == 'offensive_force_buildings':
        #     #self.offensive_force_buildings
        #     self.score_data[17] += 1

    # attempt to fix workers starting the warp in of a building
    # and not going back to work until its finished.
    # checks for idle workers then calls a distribute_workers
    # to send them back to work.
    # Does not work on workers who create assimilators since they're
    # being assigned to get gas upon starting the build
    async def back_to_work(self):
        #print('back_to_work')
        if self.idle_worker_count > 0:
            self.distribute_workers

    def find_target(self, state):
            if len(self.known_enemy_units) > 0:
                return random.choice(self.known_enemy_units)
            elif len(self.known_enemy_structures) > 0:
                return random.choice(self.known_enemy_structures)
            else:
                return self.enemy_start_locations[0]
                
    # Action 1 - Attack
    async def attack(self):
        print('attack')
        if self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, COLOSSUS]).amount > 6:
            for s in self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, COLOSSUS]).idle:
                await self.do(s.attack(self.find_target(self.state))) 

    # Action 2 - build assimilators
    # TODO
    # need to add check to move probes onto gas at this same step
    async def build_assimilators(self):
        print('build_assimilators')
        if self.supply_cap > 16:
            for nexus in self.units(NEXUS).ready:
                vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
                for vaspene in vaspenes:
                    if not self.can_afford(ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vaspene.position)
                    if worker is None:
                        break
                    if not self.units(ASSIMILATOR).closer_than(
                                                    1.0, vaspene).exists:
                        await self.do(worker.build(ASSIMILATOR, vaspene))

    # TODO
    # May need to save for output what types of units its creating for learning purposes
    # Also could limit it as was done by others
    
    # Action 3 - build offensive force
    async def build_offensive_force(self):
        print('build_offensive_force')
        # updates variables that indicate if a building exists
        # used to check if a unit can be built
        if self.units(GATEWAY).ready.exists:
            print('gateway exists', self.units(GATEWAY).ready.exists)
            GATEWAY_IND = 1
        else:
            GATEWAY_IND = 0

        if self.units(CYBERNETICSCORE).ready.exists:
            print('cyber exists', self.units(CYBERNETICSCORE).ready.exists)
            CYBERCORE_IND = 1
        else:
            CYBERCORE_IND = 0

        if self.units(ROBOTICSFACILITY).ready.exists:
            print('robo-fac exists', self.units(ROBOTICSFACILITY).ready.exists)
            ROBOFACILITY_IND = 1
        else:
            ROBOFACILITY_IND = 0

        if self.units(STARGATE).ready.exists:
            print('stargate exists', self.units(STARGATE).ready.exists)
            STARGATE_IND = 1
        else:
            STARGATE_IND = 0

        if self.units(ROBOTICSBAY).ready.exists:
            print('robo-bay exists', self.units(ROBOTICSBAY).ready.exists)
            ROBOBAY_IND = 1
        else:
            ROBOBAY_IND = 0

        # random choice of what unit to build
        # limited by the buildings that unlock the unit being built
        if ROBOBAY_IND == 1 and ROBOFACILITY_IND == 1:
            print('random 1-6')
            unit_choice = unit_list[random.randint(1, 6)]

        elif ROBOFACILITY_IND == 1 and STARGATE_IND == 1:
            print('random 1-5')
            unit_choice = unit_list[random.randint(1, 5)]

        elif ROBOFACILITY_IND == 1 and STARGATE_IND == 0:
            print('random 1-4')
            unit_choice = unit_list[random.randint(1, 4)]

        elif CYBERCORE_IND == 1:
            print('random 1-3')
            unit_choice = unit_list[random.randint(1, 3)]

        elif GATEWAY_IND == 1:
            print('zealot')
            unit_choice = unit_list[1]

        else:
            print('none')
            unit_choice = unit_list[0]

        # TODO
        # Change the check from .ready.idle to a count method
        # currently only queues one unit at a time
        if unit_choice == 'ZEALOT' and self.can_afford(ZEALOT) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(ZEALOT))
        
        elif unit_choice == 'STALKER' and self.can_afford(STALKER) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(STALKER))

        elif unit_choice == 'ADEPT' and self.can_afford(ADEPT) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(ADEPT))

        elif unit_choice == 'IMMORTAL' and self.can_afford(IMMORTAL) and \
        self.supply_left >= 4:
            for gw in self.units(ROBOTICSFACILITY).ready.idle:
                await self.do(gw.train(IMMORTAL))

        elif unit_choice == 'VOIDRAY' and self.can_afford(VOIDRAY) and \
        self.supply_left >= 4:
            for gw in self.units(STARGATE).ready.idle:
                await self.do(gw.train(VOIDRAY))

        elif unit_choice == 'COLOSSUS' and self.can_afford(COLOSSUS) and \
        self.supply_left >= 6:
            for gw in self.units(ROBOTICSFACILITY).ready.idle:
                await self.do(gw.train(COLOSSUS))

    # async def do_nothing(self):
    #     print('do_nothing')
    #     FUNCTIONS.no_op()
    #     wait = random.randrange(7,100)/100
    #     self.do_something_after = self.time + wait

    async def do_nothing(self):
        #print('do_nothing')
        wait = random.randrange(10, 30)/100
        self.do_something_after = self.time_loop + wait

    # builds 16 workers per nexus up to a maximum of 50
    async def build_workers(self):
        print('build_workers')
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and \
                                           len(self.units(PROBE)) \
                                           < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.idle:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))


    async def build_pylons(self):
        print('build_pylons')
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    # This may be an issue, watch if they only build at starting nexus
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))


    async def expand(self):
        print('expand')
        if self.units(NEXUS).amount < \
          (self.time / self.time) and (
            self.can_afford(NEXUS)) and (
            # Added to try to prevent creating multiple nexus next to each other on expansion
            not self.already_pending(NEXUS)):
            await self.expand_now()


    async def offensive_force_buildings(self):
        print('offensive_force_buildings')
        # Checks for a pylon as an indicator of where to build
        # small area around pylon is needed to place another building
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            # Gateway required first
            if self.can_afford(GATEWAY) and not \
                self.already_pending(GATEWAY) and not \
                self.units(GATEWAY).amount <= 2:
                await self.build(GATEWAY, near=pylon)

            if self.units(GATEWAY).ready.exists and not \
               self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not \
                   self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not \
                    self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def smart_action(self):
        if self.time_loop > self.do_something_after:
            choice = random.randrange(0, 9)
            # if self.use_model:
            #     prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
            #     choice = np.argmax(prediction[0])
            # else:
            #     choice = random.randrange(0, 9)
            print(self.actions[choice])
            try:
                await self.actions[choice]()
            except Exception as e:
                print(str(e))
            y = np.zeros(10)
            y[choice] = 1
            self.actions_data.append([y, self.score_data])

def main():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, BinaryBot()),
        Computer(Race.Terran, Difficulty.VeryEasy)
        ], realtime=True)

if __name__ == '__main__':
    main()

#def main():
#
#    # depending on the number selected a difficulty is chosen
#    if diff == 0:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.VeryEasy)
#            ], realtime=False)
#
#    if diff == 1:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Easy)
#            ], realtime=False)
#    
#    if diff == 2:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Medium)
#            ], realtime=False)
#
#    if diff == 3:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.MediumHard)
#            ], realtime=False)
#    
#    if diff == 4:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Hard)
#            ], realtime=False)
#
#    if diff == 5:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Harder)
#            ], realtime=False)
#    
#    if diff == 6:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.VeryHard)
#            ], realtime=False)
#
#    if diff == 7:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatVision)
#            ], realtime=False)
#
#    if diff == 8:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatMoney)
#            ], realtime=False)
#
#    if diff == 9:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatInsane)
#            ], realtime=False)
#
#if __name__ == '__main__':
#    main()