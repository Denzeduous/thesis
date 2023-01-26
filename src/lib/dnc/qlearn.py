import chess
import math
import random
import torch
from torch import optim
from torch.nn import BCELoss
import numpy as np
from gymnasium import Env
from dnc import DNC
from lib.dnc.agent import DNCChessAgent
from keras.layers import Softmax
from collections import deque
import pandas as pd
from pandas.core.common import flatten

class DNCQLearnAgent():
	def __init__(self, model: DNC, chess_agent: DNCChessAgent,
	             name: str, env: Env, state_size: int, episodes: int,
	             learn_rate: float = 0.001, gamma: float = 0.95,
	             epsilon: float = 1.0, epsilon_min: float = 0.01,
	             epsilon_decay: float = 0.99999, max_mem: int = 2_000):
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

		self._loss = BCELoss()
		self._optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

	def step(self, env, state, dnc_state):
		'''
			Q-Learning step with randomness based on epsilon.
		'''
		self.epsilon *= self.epsilon_decay
		
		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		if np.random.uniform(0, 1) <= self.epsilon:
			return np.random.choice(self.env.possible_actions)

		move = self.chess_agent.get_move_training(state, dnc_state)

		if move not in self.env.possible_actions:
			raise Exception(f'INVALID MOVE! {move} in set {self.env.possible_actions}')
			return self.env.possible_actions[random.randrange(len(self.env.possible_actions))]

		return move

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

		for state, action, reward, next_state, terminal in sample_batch:
			dnc_state = self.model.reset()

			prediction = None

			with torch.no_grad():
				prediction = np.array(self.model(torch.tensor(next_state), dnc_state))

			print(prediction)
			target = reward + self.gamma * np.amax(prediction) * bool(terminal)
			target_sample = None

			with torch.no_grad():
				out, state = self.model(torch.tensor(state), dnc_state)
				target_sample = np.array(out)
			print(target_sample)

			loss = _loss(torch.sigmoid(torch.stack(target_sample))).transpose(0, 1)
			loss.backward()
			
			target_sample[0][0] = target
			print(target_sample[0][0])

			self._optimizer.zero_grad()
			self._optimizer.step()

			loss.append(loss.item())

		self.accuracy[self.episode] = np.average(accuracy)
		self.loss[self.episode] = np.average(loss)

		# Decay the exploration
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		self.episode += 1