import chess
import math
import random
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import numpy as np
from gymnasium import Env
from dnc import DNC
from lib.dnc import ChessAgent
from keras.layers import Softmax
from collections import deque
import pandas as pd
from pandas.core.common import flatten

class QLearnAgent():
	def __init__(self, model: DNC, chess_agent: ChessAgent,
	             name: str, env: Env, state_size: int, episodes: int,
	             learn_rate: float = 0.001, gamma: float = 0.95,
	             epsilon: float = 1.0, epsilon_min: float = 0.05,
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

		self._loss = CrossEntropyLoss()
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
		
		return np.array(list(flatten(state.values()))).reshape(1, self.state_size)

	def save_model(self):
		with open(f'{self.name}.pth.tar', 'wb+') as file:
			torch.save(self.model, file)

	def step(self, env, state, dnc_state):
		'''
			Q-Learning step with randomness based on epsilon.
		'''
		self.epsilon *= self.epsilon_decay
		
		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		if np.random.uniform(0, 1) <= self.epsilon:
			return np.random.choice(self.env.possible_actions), dnc_state, None, None

		move, dnc_state, pred_from, pred_to = self.chess_agent.get_move_training(state, dnc_state)

		if move not in self.env.possible_actions:
			raise Exception(f'INVALID MOVE! {move} in set {self.env.possible_actions}')
			return self.env.possible_actions[random.randrange(len(self.env.possible_actions))]

		return move, dnc_state, pred_from, pred_to

	def remember(self, state, action, reward, next_state, terminal):
		'''
			Store the state and next state in the deque to sample later.
		'''
		state = self.reform_state(state).reshape(1, self.state_size)
		next_state = self.reform_state(next_state).reshape(1, self.state_size)

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

			prediction, dnc_state = self.model(torch.tensor(next_state), dnc_state)

			target = np.argmax(prediction.detach().numpy()[0]) * bool(terminal)
			target_sample = None

			with torch.no_grad():
				target_sample, dnc_state = self.model(torch.tensor(state), dnc_state)
				target_sample[0, target] = reward / 200 + self.gamma * prediction[0][target]

			model_loss = self._loss(torch.tanh(prediction), torch.tanh(target_sample))
			model_loss.backward()

			self._optimizer.zero_grad()
			self._optimizer.step()

			loss.append(model_loss.detach().numpy())

		self.loss[self.episode] = np.average(loss)

		# Decay the exploration
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		self.episode += 1