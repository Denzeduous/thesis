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

def main():
	'''
		Temporary testing.
	'''
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder='main', reward_type=None)
	
	dnn_model = load_model('ChessDNN.h5')
	dnn_agent = dnn.ChessAgent(dnn_model, env)

	dnc_model = torch.load('ChessDNC.pth.tar')
	dnc_agent = dnc.ChessAgent(dnc_model, env)
	dnc_state = dnc_model.reset()

	state = env.reset()

	terminated = False
	white = True

	while not bool(terminated):
		move = None
		pred_from = None
		pred_to = None

		if white:
			move, pred_from, pred_to = dnn_agent.get_move(state)
		else:
			print('DNC DNC DNC DNC DNC DNC')
			move, dnc_state, pred_from, pred_to = dnc_agent.get_move(state, dnc_state)

		white = not white

		next_state, reward, terminated, truncated, info = env.step(move, pred_from=pred_from, pred_to=pred_to)

		print(terminated)
		print(info)

		env.render()

		state = next_state
		
	env.render()

if __name__ == '__main__':
	main()