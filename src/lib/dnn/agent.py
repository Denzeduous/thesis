import chess
import keras
import random
import numpy as np
import gymnasium as gym
from keras.layers import Softmax
from pandas.core.common import flatten

class ChessAgent:
	def __init__(self, model: keras.Model, env: gym.Env, states: int):
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
			return state.reshape(1, 1, self.state_size)

		if isinstance(state, tuple):
			state = state[0] # No idea why this happens tbh

		if not isinstance(state['board'], np.ndarray):
			state['board'] = np.concatenate(state['board'])
		
		#if not isinstance(state['ownership'], np.ndarray):
			#state['ownership'] = np.concatenate(state['ownership'])

		if not isinstance(state['player'], int):
			state['player'] = 0 if state['player'] == 'White' else 1
		
		return np.array(list(flatten(state.values()))).reshape(1, 1, self.state_size)

	def get_move_training(self, state):
		state = self.reform_state(state)

		actions = self.model.predict(state, verbose=0)[0][0]

		# Get the probability subsets
		probability_from = actions[  :64]
		probability_to   = actions[64:-4]
		probability_pro  = actions[-4:  ]

		# print(probability_from)
		# print(probability_to)
		# print(probability_pro)

		prob_from_loc = np.argsort(probability_from)[::-1]
		prob_to_loc   = np.argsort(probability_to)[::-1]
		prob_pro_loc  = np.argsort(probability_pro)[::-1]

		from_squares = [move.from_square for move in self.env.possible_actions]
		loc_from = [loc for loc in prob_from_loc if loc in from_squares]

		idx_from = None

		for loc in range(len(loc_from)):
			idx_from = loc_from[loc]
			
			if random.uniform(0, 1) >= 0.5:
				break

		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]
		loc_to = [loc for loc in prob_to_loc if loc in to_squares]

		idx_to = None

		for loc in range(len(loc_to)):
			idx_to = loc_to[loc]
			
			if random.uniform(0, 1) >= 0.5:
				break

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]
		loc_pro = [loc for loc in prob_pro_loc if loc in promotions]

		promotion = None

		if len(promotions) > 0:
			for loc in range(len(loc_pro)):
				promotion = loc_pro[loc] + 2 # The actual promotion values are 2 off

				if random.uniform(0, 1) >= 0.5:
					break

		probability_from += abs(np.min(probability_from))
		probability_to   += abs(np.min(probability_to))

		return chess.Move(idx_from, idx_to, promotion), probability_from / sum(probability_from), probability_to / sum(probability_to)

	def get_move(self, state):
		state = self.reform_state(state)
		actions = self.model.predict(state, verbose=0)[0][0]

		# Get the probability subsets and sort them
		from_arr = np.argsort(np.array(actions[  :64]))[::-1]
		to_arr   = np.argsort(np.array(actions[64:-4]))[::-1]
		pro_arr  = np.argsort(np.array(actions[-4:  ]))[::-1]

		from_squares = [move.from_square for move in self.env.possible_actions]
		loc_from = [loc for loc in from_arr if loc in from_squares]
		idx_from = loc_from[0]

		if idx_from == None: raise Exception(f'Unknown error in from_squares {from_squares}.')
		
		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]
		loc_to = [loc for loc in to_arr if loc in to_squares]
		idx_to = loc_to[0]

		if idx_to == None: raise Exception(f'All were None in to_squares {to_squares}.')
		
		promotion = None

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]

		if len(promotions) > 0:
			loc_pro = [loc for loc in pro_arr if loc in promotions]
			promotion = loc_pro[0] + 2

		# Get the probability subsets
		probability_from = actions[  :64]
		probability_to   = actions[64:-4]

		return chess.Move(idx_from, idx_to, promotion), probability_from / sum(probability_from), probability_to / sum(probability_to)