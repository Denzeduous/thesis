import chess
import keras
import random
import numpy as np
import gymnasium as gym
from keras.layers import Softmax
from pandas.core.common import flatten

class ChessAgent:
	def __init__(self, model: keras.Model, env: gym.Env):
		self.model = model
		self.env   = env
		self.board = env.board
		self.state_size = env.observation_space['board'].n + env.observation_space['player'].n

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

		prob_from_loc = np.argsort(probability_from)
		prob_to_loc   = np.argsort(probability_to)
		prob_pro_loc  = np.argsort(probability_pro)

		from_squares = [move.from_square for move in self.env.possible_actions]

		# This code works off of a "bracket" probability system.
		# Say z is the probability we wish to reach, and x + y == z.
		# x = 0.4, y = 0.55, z = 0.9.
		# 
		# We go through each of the two probabilities and add them together, checking each time.
		# x = 0.4, which is less than 0.9.
		# y = 0.55, which x + y > z, thus, this is the value we should choose.

		i = 0
		bracket = 0
		rand = random.uniform(0, 1)
		idx_from = None
		fallback = None

		for probability in prob_from_loc:
			if bracket + probability_from[probability] > rand and i in from_squares:
				idx_from = i
				break

			if fallback == None and i in from_squares:
				fallback = i
			
			bracket += probability_from[probability]
			i += 1

		if idx_from == None: idx_from = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]

		i = 0
		bracket = 0
		rand = random.uniform(0, 1)
		idx_to = None
		fallback = None

		for probability in prob_to_loc:
			if bracket + probability_to[probability] > rand and i in to_squares:
				idx_to = i
				break

			if fallback == None and i in to_squares:
				fallback = i
			
			bracket += probability_to[probability]
			i += 1

		if idx_to == None: idx_to = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		i = 0
		bracket = 0
		rand = random.uniform(0, 1)
		promotion = None
		fallback = None

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]

		if len(promotions) > 0:
			for probability in prob_pro_loc:
				if bracket + probability_pro[probability] > rand and i in promotions:
					promotion = i
					break

				if fallback == None and i in promotions:
					fallback = i
				
				bracket += probability_pro[probability]
				i += 1

			if promotion == None: promotion = fallback # This can happen if the sum of probabilities < 1.0. Can occur

		return chess.Move(idx_from, idx_to, promotion), probability_from, probability_to

	def get_move(self, state):
		state = self.reform_state(state)
		actions = self.model.predict(state, verbose=0)[0][0]

		# Get the probability subsets and sort them
		from_arr = np.argsort(np.array(actions[  :64]))[::-1]
		to_arr   = np.argsort(np.array(actions[64:-4]))[::-1]
		pro_arr  = np.argsort(np.array(actions[-4:  ]))[::-1]

		from_squares = [move.from_square for move in self.env.possible_actions]
		idx_from = None

		for idx in from_arr:
			if idx in from_squares:
				idx_from = int(idx)
				break

		if idx_from == None: raise Exception(f'Unknown error in from_squares {from_squares}.')
		
		to_squares = [move.to_square for move in self.env.possible_actions if move.from_square == idx_from]

		idx_to = None

		for idx in to_arr:
			if idx in to_squares:
				idx_to = int(idx)
				break

		if idx_to == None: raise Exception(f'All were None in to_squares {to_squares}.')
		
		promotion = None

		promotions = [move.promotion for move in self.env.possible_actions if move.from_square == idx_from and move.promotion != None]

		if len(promotions) > 0:
			for idx in pro_arr:
				if idx in promotions:
					promotion = int(idx)
					break

		# Get the probability subsets
		probability_from = actions[  :64]
		probability_to   = actions[64:-4]

		return chess.Move(idx_from, idx_to, promotion), probability_from, probability_to