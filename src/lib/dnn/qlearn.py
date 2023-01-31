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
	             epsilon: float = 1.0, epsilon_min: float = 0.05,
	             epsilon_decay: float = 0.999999, max_mem: int = 2_000):
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
		self.accuracy = np.zeros(episodes)
		self.loss = np.zeros(episodes)
		self.episode = 0

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
		
		if not isinstance(state['player'], int):
			state['player'] = 0 if state['player'] == 'White' else 1
		
		return np.array(list(flatten(state.values()))).reshape(1, 1, self.state_size)

	def save_model(self):
		self.model.save(self.name + '.h5')

	def step(self, env, state):
		'''
			Q-Learning step with randomness based on epsilon.
		'''
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

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

		sample_batch = random.sample(self.memory, sample_batch_size)

		accuracy = []
		loss = []
		i = 1

		for state, action, reward, next_state, terminal in sample_batch:
			prediction = self.model.predict(next_state, verbose=0)

			actions = prediction[0][0]

			# Get the probability subsets
			probability_from = actions[  :64]
			probability_to   = actions[64:-4]
			probability_pro  = actions[-4:  ]

			target_from = np.argmax(probability_from).item()
			target_to   = np.argmax(probability_to  ).item()
			target_pro  = np.argmax(probability_pro ).item()

			target_sample = self.model.predict(state, verbose=0)

			target_sample[0][0][target_from] = reward / 200 + self.gamma * prediction[0][0][target_from]
			target_sample[0][0][target_to]   = reward / 200 + self.gamma * prediction[0][0][target_to]
			target_sample[0][0][target_pro]  = reward / 200 + self.gamma * prediction[0][0][target_pro]

			history = self.model.fit(state, target_sample, epochs=1, verbose=0)

			accuracy.append(history.history['accuracy'])

			loss.append(history.history['loss'])

		self.accuracy[self.episode] = np.average(accuracy)
		self.loss[self.episode] = np.average(loss)

		# Decay the exploration
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		self.episode += 1