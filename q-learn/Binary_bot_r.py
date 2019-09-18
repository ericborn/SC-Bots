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
import math
import time
import csv
import os
import asyncio

# pysc2 libraries
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# sc2 libraries
import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY

# Q learning system found here:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(
                                                state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] *
                                               len(self.actions),
                                               index=self.q_table.columns,
                                               name=state))

# Creates variables holding strings for bots actions
ACTION_distribute_workers = 'distribute_workers'
ACTION_build_workers = 'build_workers'
ACTION_build_pylons = 'build_pylons'
ACTION_build_assimilators = 'build_assimilators'
ACTION_expand = 'expand'
ACTION_offensive_force_buildings = 'offensive_force_buildings'
ACTION_build_offensive_force = 'build_offensive_force'
ACTION_attack = 'attack'

# creates a list of actions the bot can choose from
smart_actions = [
    ACTION_distribute_workers,
    ACTION_build_workers,
    ACTION_build_pylons,
    ACTION_build_assimilators,
    ACTION_expand,
    ACTION_offensive_force_buildings,
    ACTION_build_offensive_force,
    ACTION_attack
]


# possible rewards or counters for the learning system
# units_destroyed = 0

# Define rewards for killing units or buildings
# KILL_UNIT_REWARD = 0.2
# KILL_BUILDING_REWARD = 0.5
# SUPPLY_ARMY_REWARD = 0.2
# SUPPLY_WORKERS_REWARD = 0.05

# KILL_UNIT_REWARD = 0.2
# KILL_BUILDING_REWARD = 0.5
# SUPPLY_ARMY_REWARD = 0.2
# SUPPLY_WORKERS_REWARD = 0.05

# Bots current issues:
# Too much expansion
# Troops need to move out to protect expanded bases
# sometimes creates multiple nexuses right next to each other
# wont target troops repairing buildings

# creates empty np array to store game stats
# and the overall game outcome
score_data = np.zeros(12)

class BinaryBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 500
        self.MAX_WORKERS = 200
        
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
            score_data[11] = -1
            self.write_csv(str(-1))
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(score_data))
        
        # Win
        elif result == 'Result.Victory':
            score_data[11] = 1
            self.write_csv(1)
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(score_data))
        
        # Tie
        else:
            score_data[11] = 0
            self.write_csv(0)
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(score_data))

    # This is the function steps forward and is called through each frame of the game
    async def on_step(self, iteration):
        self.iteration = iteration

        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()
        
        # checks to see if the value has increased, if so records new value
        # supplies
        if score_data[0] < self.supply_cap:
            score_data[0] = self.supply_cap

        # TODO consider using a ratio of supply workers to supply army
        # instead of just totals
        if score_data[1] < self.supply_army:
            score_data[1] = self.supply_army

        if score_data[2] < self.supply_workers:
            score_data[2] = self.supply_workers

        # built structures
        if score_data[3] < self.units(PYLON).amount:
            score_data[3] = self.units(PYLON).amount

        if score_data[4] < self.units(ASSIMILATOR).amount:
            score_data[4] = self.units(ASSIMILATOR).amount

        if score_data[5] < self.units(GATEWAY).amount:
            score_data[5] = self.units(GATEWAY).amount

        if score_data[6] < self.units(STALKER).amount:
            score_data[6] = self.units(STALKER).amount

        if score_data[7] < self.units(VOIDRAY).amount:
            score_data[7] = self.units(VOIDRAY).amount

        if score_data[8] < self.units(NEXUS).amount:
            score_data[8] = self.units(NEXUS).amount

        # Destroyed buildings and units
        if score_data[9] < self.state.score.killed_value_structures:
            score_data[9] = self.state.score.killed_value_structures

        if score_data[10] < self.state.score.killed_value_units:
            score_data[10] = self.state.score.killed_value_units

    # TODO Convert all actions into smart actions
    # setup random selection bot to pick from these
    # ?? Possibly create a counter to record the number each action is taken ??
    async def build_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and \
                                           len(self.units(PROBE)) \
                                           < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
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

    async def expand(self):
        if self.units(NEXUS).amount < \
          (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(
                                                                 NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not \
               self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not \
                   self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif len(self.units(GATEWAY)) < \
                    ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                if self.can_afford(GATEWAY) and not \
                   self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < \
                      ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                    if self.can_afford(STARGATE) and not \
                       self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        if self.units(GATEWAY).ready.exists and \
           self.units(CYBERNETICSCORE).ready.exists:
            for gw in self.units(GATEWAY).ready.noqueue:
                if not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                    if self.can_afford(STALKER) and self.supply_left > 0:
                        await self.do(gw.train(STALKER))

        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        aggressive_units = {STALKER: [15, 5],
                            VOIDRAY: [8, 3]}

        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and \
               self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(
                                random.choice(self.known_enemy_units)))

def main():
    # Creates a random number between 0-9
    diff = random.randrange(0,10)

    # depending on the number selected a difficulty is chosen
    if diff == 0:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.VeryEasy)
            ], realtime=False)

    if diff == 1:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.Easy)
            ], realtime=False)
    
    if diff == 2:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.Medium)
            ], realtime=False)

    if diff == 3:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.MediumHard)
            ], realtime=False)
    
    if diff == 4:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.Hard)
            ], realtime=False)

    if diff == 5:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.Harder)
            ], realtime=False)
    
    if diff == 6:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.VeryHard)
            ], realtime=False)

    if diff == 7:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.CheatVision)
            ], realtime=False)

    if diff == 8:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.CheatMoney)
            ], realtime=False)

    if diff == 9:
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, BinaryBot()),
            Computer(Race.Terran, Difficulty.CheatInsane)
            ], realtime=False)

if __name__ == '__main__':
    main()
