# -*- coding: utf-8 -*-
"""
@author: Eric Born
"""
import numpy as np
import pandas as pd

from pysc2.agents import base_agent

import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random
#from s2clientprotocol import score_pb2

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
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
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
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
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

# Creates variables holding strings for bots actions
ACTION_distribute_workers = 'distribute_workers'
ACTION_build_workers = 'build_workers'
ACTION_build_pylons = 'build_pylons'
ACTION_build_assimilators = 'build_assimilators'
ACTION_expand = 'expand'
ACTION_offensive_force_buildings = 'offensive_force_buildings'
ACTION_build_offensive_force= 'build_offensive_force'
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

#### Bots current issues:
# Too much expansion
# Troops need to move out to protect expanded bases

class IronBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 500
        self.MAX_WORKERS = 200

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        #self.previous_killed_unit_score = 0
        #self.previous_killed_building_score = 0
        
        self.previous_action = None
        self.previous_state = None

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

        # updated after every action???
        # creates empty np array to store current states
        current_state = np.zeros(20)

        # sets array with supply numbers
        current_state[0] = self.supply_cap
        current_state[1] = self.supply_left
        current_state[2] = self.supply_used
        current_state[3] = self.supply_army
        current_state[4] = self.supply_workers

        # # sets array with unit/building numbers
        # current_state[5] = self.units(PYLON).amount
        # current_state[6] = self.units(ASSIMILATOR).amount
        # current_state[7] = self.units(GATEWAY).amount
        # current_state[8] = self.units(STALKER).amount
        # current_state[9] = self.units(VOIDRAY).amount
        # current_state[10] = self.units(NEXUS).amount

        # # sets array with resource numbers
        # current_state[11] = self.minerals
        # current_state[12] = self.vespene
        # current_state[13] = self.geysers

        #### not sure if the bot should be able track total number of units and buildings killed, 
        # thats information a human player could certainly generalize at a high level
        # but impossible to track at 100% certainty
        # old method was obs, but unsure how that works
        # obs was brought into the step method and then referenced for finding unit and score amounts
        #killed_unit_score = obs.observation['score_cumulative'][5]
        #killed_building_score = obs.observation['score_cumulative'][6]

        # New method
        #current_state[14] = units
        #current_state[15] = buildings

        print(current_state)
        

    async def build_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and \
        len(self.units(PROBE)) < self.MAX_WORKERS:
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
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount < \
        (self.iteration / self.ITERATIONS_PER_MINUTE) and \
        self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        #print(self.iteration / self.ITERATIONS_PER_MINUTE)
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
    # added first if statement to check if gateway and cyber exist
    async def build_offensive_force(self):
        if self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).ready.exists:
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
        # {UNIT: [n to fight, n to defend]}
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


run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, IronBot()),
    Computer(Race.Terran, Difficulty.Hard)
    ], realtime=False)