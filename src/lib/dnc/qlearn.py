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
			return self.env.possible_actions[random.randrange(len(self.env.possible_actions))]

		return move, pred_from, pred_to

	def remember(self, state, action, reward, next_state, terminal):
		'''
			Store the state and next state in the deque to sample later.
		'''
		state = self.reform_state(state).reshape(1, self.state_size)
		next_state = self.reform_state(next_state).reshape(1, self.state_size)

		# They swap after a step
		states1 = self.model._states2
		states2 = self.model._states

		self.memory.append((state, action, reward, next_state, terminal, states1, states2))

	def replay(self, sample_batch_size):
		'''
			Replay the memories and train the model.
		'''
		if len(self.memory) < sample_batch_size: return

		sample_batch = random.sample(self.memory, sample_batch_size)

		loss = []

		for state, action, reward, next_state, terminal, states1, states2 in sample_batch:
			self.model.reset()
			self.model._states  = states1
			self.model._states2 = states2

			prediction = self.model(torch.tensor(next_state))

			target_sample = None

			with torch.no_grad():
				actions = prediction[0]

				target = 1 / (1 + np.exp(-reward))

				if terminal:
					target *= 5
				else:
					target += self.gamma * np.amax(prediction[0].numpy())

				target_sample = self.model(torch.tensor(state))

				target_sample[0, action.from_square]    = target
				target_sample[0, action.to_square + 64] = target

				if action.promotion != None:
					target_sample[0, action.promotion] = reward

			model_loss = self._loss(torch.tanh(prediction), torch.tanh(target_sample))
			
			self._optimizer.zero_grad()
			
			try:
				model_loss.backward()

				self._optimizer.step()
			except Exception as e:
				pass

			loss.append(model_loss.detach().numpy())

		self.loss[self.episode] = np.average(loss)

		# Decay the exploration
		self.epsilon *= self.epsilon_decay

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min

		self.episode += 1