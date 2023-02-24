import chess
import math
import random
import numpy as np
from lib.dnn import ChessAgent
from gymnasium import Env
from keras import Model
from keras.layers import Softmax
from collections import deque
import pandas as pd
from pandas.core.common import flatten

class QLearnAgent():
	def __init__(self, model: Model, chess_agent: ChessAgent,
				 name: str, env: Env, state_size: int, episodes: int,
				 learn_rate: float = 0.001, gamma: float = 0.95,
				 epsilon: float = 1.0, epsilon_min: float = 0.1,
				 epsilon_decay: float = 0.99999, tau: int = 50,
				 max_mem: int = 2_000):
		self.model = model
		self.chess_agent = chess_agent
		self.name = name
		self.env = env
		self.state_size = state_size
		self.memory = deque(maxlen = max_mem)
		self.learn_rate = learn_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.tau = tau
		self.accuracy = np.zeros(episodes)
		self.loss = np.zeros(episodes)
		self._weights = model.get_weights()
		self._episode = 1

	def reform_state(self, state):
		'''
			Reforms the state into a flattened 1D array.
			
			The `isinstance` calls within are to make sure
			that it hasn't been called before on the same state.
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

	def save_model(self):
		self.model.save(self.name + '.h5')

	def step(self, env, state):
		'''
			Q-Learning step with randomness based on epsilon.
		'''
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

		if np.random.uniform(0, 1) <= self.epsilon:
			return np.random.choice(self.env.possible_actions), None, None

		move, pred_from, pred_to = self.chess_agent.get_move_training(state)

		if move not in self.env.possible_actions:
			raise Exception(f'INVALID MOVE! {move} in set {self.env.possible_actions}')

		return move, pred_from, pred_to

	def remember(self, state, action, reward, next_state, terminal):
		'''
			Store the state and next state in the deque to sample later.
		'''
		state = self.reform_state(state).reshape(1, 1, self.state_size)
		next_state = self.reform_state(next_state).reshape(1, 1, self.state_size)
		self.memory.append((state, action, reward, next_state, terminal))

	def replay(self, sample_batch_size):
		'''
			Replay the memories and train the model.
		'''
		if len(self.memory) < sample_batch_size: return

		if self._episode % self.tau == 0:
			self._weights = self.model.get_weights()

		sample_batch = random.sample(self.memory, sample_batch_size)

		accuracy = []
		loss = []

		targets = []

		# for state, action, reward, next_state, terminal in sample_batch:
		# 	prediction = self.model.predict(next_state, verbose=0)

		# 	actions = prediction[0][0]

		# 	reward = 1 / (1 + np.exp(-reward))

		# 	target_sample = self.model.predict(state, verbose=0)

		# 	target_sample[0][0][action.from_square]    = reward + self.gamma * np.amax(prediction[0][0])
		# 	target_sample[0][0][action.to_square + 64] = reward + self.gamma * np.amax(prediction[0][0])
			
		# 	if action.promotion != None:
		# 		target_sample[0][0][action.promotion] = reward / 2_000 + self.gamma * np.amax(prediction[0][0])

		# 	targets.append(target_sample[0])

		# history = self.model.fit(np.array([state[0] for state, _, _, _, _ in sample_batch]), np.array(targets), epochs=1, verbose=0)

		states, targets = [], []

		for state, action, reward, next_state, done in sample_batch:
			prediction = self.model.predict(next_state)

			target_from = reward + self.gamma * np.amax(prediction[0][0][  :64])
			target_to   = reward + self.gamma * np.amax(prediction[0][0][64:-4])
			target_pro  = reward + self.gamma * np.amax(prediction[0][0][  :-4])

			weights = self.model.get_weights()
			self.model.set_weights(self._weights)

			target_sample = self.model.predict(state)
			target_sample[0][0][action.from_square]    = target_from 
			target_sample[0][0][action.to_square + 64] = target_to

			if action.promotion != None:
				target_sample[0][0][action.promotion + 128 - 2] = target_pro

			self.model.set_weights(weights)

			# Batch training to make things faster
			states.append(state[0])
			targets.append(target_sample[0])

		history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
		# Keeping track of loss

		accuracy.append(history.history['accuracy'])
		loss.append(history.history['loss'])

		self.accuracy[self._episode - 1] = np.average(accuracy)
		self.loss[self._episode - 1] = np.average(loss)

		# Decay the exploration
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		self._episode += 1