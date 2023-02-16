import chess
import keras
import random
import torch
import math
import numpy as np
import gymnasium as gym
from .model import SequentialDNC
from pandas.core.common import flatten

class ChessAgent:
	def __init__(self, model: SequentialDNC, env: gym.Env, states: int):
		self.model = model
		self.env   = env
		self.board = env.board
		self.state_size = states

	def reform_state(self, state):
		'''
			Reforms the state into a flattened 1D array.
			
			The `isinstance` calls within are to make sure
			that it hasn't been called before on the same state.

			Unfortunately, this has to be copied from the QLearn.
		'''
		if isinstance(state, np.ndarray):
			return state.reshape(1, self.state_size)

		if isinstance(state, tuple):
			state = state[0] # No idea why this happens tbh

		if not isinstance(state['board'], np.ndarray):
			state['board'] = np.concatenate(state['board'])

		if not isinstance(state['player'], int):
			state['player'] = 0 if state['player'] == 'White' else 1

		return np.array(list(flatten(state.values()))).reshape(1, self.state_size)

	def get_move_training(self, state):
		state = self.reform_state(state)
		actions = None

		with torch.no_grad():
			actions = self.model(torch.tensor(state))

		actions = actions[0]

		# Get the probability subsets
		probability_from = actions[  :64]
		probability_to   = actions[64:-4]
		probability_pro  = actions[-4:  ]

		prob_from_loc = torch.flip(np.argsort(probability_from), dims=(0,))
		prob_to_loc   = torch.flip(np.argsort(probability_to),   dims=(0,))
		prob_pro_loc  = torch.flip(np.argsort(probability_pro),  dims=(0,))

		from_squares = [move.from_square for move in self.env.possible_actions]
		loc_from = [loc for loc in prob_from_loc if loc in from_squares]

		bracket = 0
		idx_from = None
		fallback = None
		rand = random.uniform(0, 0.9)

		for loc in range(len(loc_from)):
			bracket += random.uniform(0, 0.15)

			if bracket > rand:
				idx_from = loc_from[loc].item()
				break

			if fallback == None:
				fallback = loc_from[loc].item()

		if idx_from == None: idx_from = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]
		loc_to = [loc for loc in prob_to_loc if loc in to_squares]

		bracket = 0
		idx_to = None
		fallback = None
		rand = random.uniform(0, 0.9)

		for loc in range(len(loc_to)):
			bracket += random.uniform(0, 0.15)

			if bracket > rand:
				idx_to = loc_to[loc].item()
				break

			if fallback == None:
				fallback = loc_to[loc].item()

		if idx_to == None: idx_to = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]
		loc_pro = [loc for loc in prob_pro_loc if loc in promotions]

		bracket = 0
		promotion = None
		fallback = None
		rand = random.uniform(0, 0.9)

		if len(promotions) > 0:
			for loc in range(len(loc_pro)):
				bracket += random.uniform(0, 0.15)
				
				if bracket > rand:
					promotion = loc_pro[loc] + 2 # The actual promotion values are 2 off
					break

				if fallback == None:
					fallback = loc_pro[loc] + 2 # The actual promotion values are 2 off

			if promotion == None: promotion = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		probability_from  = probability_from.numpy()
		probability_from += abs(np.min(probability_from))
		probability_to  = probability_to.numpy()
		probability_to += abs(np.min(probability_to))

		return chess.Move(idx_from, idx_to, promotion), probability_from / sum(probability_from), probability_to / sum(probability_to)

	def get_move(self, state):
		state = self.reform_state(state)
		actions = None

		with torch.no_grad():
			actions = self.model(torch.tensor(state))

		actions = actions[0]

		from_squares = [move.from_square for move in self.env.possible_actions]
		loc_from = [loc for loc in prob_from_loc if loc in from_squares]
		idx_from = loc_from[0].item()

		if idx_from == None: raise Exception(f'Unknown error in from_squares {from_squares}.')
		
		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]
		loc_to = [loc for loc in prob_to_loc if loc in to_squares]
		idx_to = loc_to[0].item()

		if idx_to == None: raise Exception(f'All were None in to_squares {to_squares}.')
		
		promotion = None

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]

		if len(promotions) > 0:
			loc_pro = [loc for loc in prob_pro_loc if loc in promotions]
			promotion = loc_pro[0] + 2

		# Get the probability subsets
		probability_from = actions[  :64]
		probability_to   = actions[64:-4]

		return chess.Move(idx_from, idx_to, promotion), probability_from, probability_to