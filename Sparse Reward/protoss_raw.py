# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:42:58 2019

@author: Eric Born
"""

import random
import math

import numpy as np
import pandas as pd

from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

#########
# Actions section
#########
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_BUILD_NEXUS = actions.FUNCTIONS.Build_Nexus_screen.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_ASSIMILATOR = actions.FUNCTIONS.Build_Assimilator_screen.id

_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

# Terrain units
#_TERRAN_COMMANDCENTER = 18
#_TERRAN_SCV = 45 
#_TERRAN_SUPPLY_DEPOT = 19
#_TERRAN_BARRACKS = 21

_PROTOSS_NEXUS = 59
_PROTOSS_PYLON = 60
_PROTOSS_ASSIMILATOR = 61
_PROTOSS_GATEWAY = 62
_PROTOSS_STALKER = 74
_PROTOSS_PROBE = 84

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_PROBE = 'selectprobe'
ACTION_BUILD_ASSIMILATOR = 'buildassimilator'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_SELECT_GATEWAY = 'selectgateway'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_BUILD_STALKER = 'buildstalker'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_PROBE,
    ACTION_BUILD_PYLON,
    ACTION_BUILD_ASSIMILATOR,
    ACTION_BUILD_GATEWAY,
    ACTION_SELECT_GATEWAY,
    ACTION_BUILD_ZEALOT,
    ACTION_BUILD_STALKER,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

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
##########

class RawAgent(base_agent.BaseAgent):
  def __init__(self):
    super(RawAgent, self).__init__()
    self.base_top_left = None

  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
  
  def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]
    
  def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
  
  def step(self, obs):
    super(RawAgent, self).step(obs)
    
    if obs.first():
      nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)[0]
      self.base_top_left = (nexus.x < 32)
    
    pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
    completed_pylons = self.get_my_completed_units_by_type(
        obs, units.Protoss.Pylon)
    
    gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
    completed_gateways = self.get_my_completed_units_by_type(
        obs, units.Protoss.Gateway)
    
    free_supply = (obs.observation.player.food_cap - 
               obs.observation.player.food_used)

    zealots = self.get_my_units_by_type(obs, units.Protoss.Zealot)
    
    if len(pylons) == 0 and obs.observation.player.minerals >= 100:
      probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
      if len(probes) > 0:
        pylon_xy = (22, 20) if self.base_top_left else (35, 42)
        distances = self.get_distances(obs, probes, pylon_xy)
        probe = probes[np.argmin(distances)]
        return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
      
    if (len(completed_pylons) > 0 and len(gateways) == 0 and 
        obs.observation.player.minerals >= 150):
      probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
      if len(probes) > 0:
        gateway_xy = (22, 24) if self.base_top_left else (35, 45)
        distances = self.get_distances(obs, probes, gateway_xy)
        probe = probes[np.argmin(distances)]
        return actions.RAW_FUNCTIONS.Build_Gateway_pt(
            "now", probe.tag, gateway_xy)
      
    if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100
        and free_supply >= 2):
      gateway = gateways[0]
      if gateway.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)
      
    if free_supply < 2 and len(zealots) > 0:
      attack_xy = (38, 44) if self.base_top_left else (19, 23)
      distances = self.get_distances(obs, zealots, attack_xy)
      zealot = zealots[np.argmax(distances)]
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
         "now", zealot.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

    # This section contains code for giving rewards for killing units or
    # buildings. needs to be updated for protoss and folded into current bot 
    # step method
#    supply_limit = obs.observation['player'][4]
#    army_supply = obs.observation['player'][5]
#    
#    killed_unit_score = obs.observation['score_cumulative'][5]
#    killed_building_score = obs.observation['score_cumulative'][6]
#    
#    ######!!!! Need up update to match protoss actions
#    current_state = np.zeros(20)
#    current_state[0] = supply_depot_count
#    current_state[1] = barracks_count
#    current_state[2] = supply_limit
#    current_state[3] = army_supply
#
#    hot_squares = np.zeros(16)        
#    enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
#    for i in range(0, len(enemy_y)):
#        y = int(math.ceil((enemy_y[i] + 1) / 16))
#        x = int(math.ceil((enemy_x[i] + 1) / 16))
#        
#        hot_squares[((y - 1) * 4) + (x - 1)] = 1
#    
#    if not self.base_top_left:
#        hot_squares = hot_squares[::-1]
#    
#    for i in range(0, 16):
#        current_state[i + 4] = hot_squares[i]
#
#    if self.previous_action is not None:
#        reward = 0
#            
#        if killed_unit_score > self.previous_killed_unit_score:
#            reward += KILL_UNIT_REWARD
#                
#        if killed_building_score > self.previous_killed_building_score:
#            reward += KILL_BUILDING_REWARD
#            
#        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
#    
#    rl_action = self.qlearn.choose_action(str(current_state))
#    smart_action = smart_actions[rl_action]
#    
#    self.previous_killed_unit_score = killed_unit_score
#    self.previous_killed_building_score = killed_building_score
#    self.previous_state = current_state
#    self.previous_action = rl_action
#
    return actions.RAW_FUNCTIONS.no_op()


def main(unused_argv):
  agent = RawAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="Simple64",
          players=[sc2_env.Agent(sc2_env.Race.protoss), 
                   sc2_env.Bot(sc2_env.Race.protoss, 
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              action_space=actions.ActionSpace.RAW,
              use_raw_units=True,
              raw_resolution=64,
          ),
      ) as env:
        run_loop.run_loop([agent], env)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)