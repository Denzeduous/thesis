import gymnasium as gym
import chess
import chess.engine
import numpy as np
import random
import os
import torch
from math import floor
from keras.layers import Softmax
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from pandas.core.common import flatten

import lib.dnn as dnn
import lib.dnc as dnc
import lib.gymchess

ANALYSIS_DEPTH = 20

def reform_state_dnn(state):
	'''
		Reforms the state into a flattened 1D array.
		
		The `isinstance` calls within are to make sure
		that it hasn't been called before on the same state.
	'''
	if isinstance(state, np.ndarray):
		return state.reshape(1, 1, 65)

	if isinstance(state, tuple):
		state = state[0] # No idea why this happens tbh

	if not isinstance(state['board'], np.ndarray):
		state['board'] = np.concatenate(state['board'])
	
	if not isinstance(state['player'], int):
		state['player'] = 0 if state['player'] == 'White' else 1
	
	return np.array(list(flatten(state.values()))).reshape(1, 1, 65)

def reform_state_dnc(state):
	'''
		Reforms the state into a flattened 1D array.
		
		The `isinstance` calls within are to make sure
		that it hasn't been called before on the same state.
	'''
	if isinstance(state, np.ndarray):
		return state.reshape(1, 1, 65)

	if isinstance(state, tuple):
		state = state[0] # No idea why this happens tbh

	if not isinstance(state['board'], np.ndarray):
		state['board'] = np.concatenate(state['board'])
	
	if not isinstance(state['player'], int):
		state['player'] = 0 if state['player'] == 'White' else 1
	
	return np.array(list(flatten(state.values()))).reshape(1, 65)

def test_dnn():
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder='main', reward_type='Simple')
	
	dnn_model = load_model('ChessDNN.h5')
	dnn_agent = dnn.ChessAgent(dnn_model, env, env.observation_space['board'].n + env.observation_space['player'].n)

	terminated = False

	state = env.reset()

	while not bool(terminated):
		move = None
		pred_from = None
		pred_to = None

		move, pred_from, pred_to = dnn_agent.get_move(state)

		next_state, reward, terminated, truncated, info = env.step(move, pred_from=pred_from, pred_to=pred_to)

		env.render()
		state = next_state

	env.render()

	print(f'Done with outcome {terminated}')

def main():
	'''
		Temporary testing.
	'''
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder='main', reward_type='Simple')
	states = env.observation_space['board'].n + env.observation_space['player'].n

	dnn_model = load_model('ChessDNN.h5')
	dnn_agent = dnn.ChessAgent(dnn_model, env, states)

	dnc_model = torch.load('ChessDNC.pth.tar')
	dnc_agent = dnc.ChessAgent(dnc_model, env, states)

	state = env.reset()

	terminated = False
	white = True

	white_reward = 0
	black_reward = 0

	while not bool(terminated):
		move = None
		pred_from = None
		pred_to = None

		if white:
			move, pred_from, pred_to = dnc_agent.get_move(state)
		else:
			move, pred_from, pred_to = dnn_agent.get_move(state)

		next_state, reward, terminated, truncated, info = env.step(move, pred_from=pred_from, pred_to=pred_to)

		if white and reward != 0:
			white_reward += reward
			print(f'White got a reward of {reward}, total of {white_reward}')
		elif reward != 0:
			black_reward += reward
			print(f'Black got a reward of {reward}, total of {black_reward}')

		white = not white

		env.render()

		state = next_state
		
	env.render()

def test_dnc():
	'''
		Temporary testing.
	'''
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder='main', reward_type='Simple')
	states = env.observation_space['board'].n + env.observation_space['player'].n

	dnc_model1 = torch.load('ChessDNC.pth.tar')
	dnc_agent1 = dnc.ChessAgent(dnc_model1, env, states)

	dnc_model2 = torch.load('ChessDNC.pth.tar')
	dnc_agent2 = dnc.ChessAgent(dnc_model2, env, states)

	state = env.reset()

	terminated = False
	white = True

	white_reward = 0
	black_reward = 0

	while not bool(terminated):
		move = None
		pred_from = None
		pred_to = None

		if white:
			move, pred_from, pred_to = dnc_agent1.get_move(state)
		else:
			move, pred_from, pred_to = dnc_agent2.get_move(state)

		white = not white

		next_state, reward, terminated, truncated, info = env.step(move, pred_from=pred_from, pred_to=pred_to)

		if white and reward != 0:
			white_reward += reward
			print(f'White got a reward of {reward}, total of {white_reward}')
		elif reward != 0:
			black_reward += reward
			print(f'Black got a reward of {reward}, total of {black_reward}')

		env.render()

		state = next_state
		
	env.render()

	print(f'Done with outcome {terminated}')

if __name__ == '__main__':
	#test_dnn()
	#test_dnc()
	main()