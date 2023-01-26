import gymnasium as gym
import lib.gymchess
import chess
import chess.engine
import numpy as np
import random
import os
from math import floor
from keras.layers import Softmax
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from pandas.core.common import flatten
from lib.dnn import DNNChessAgent

ANALYSIS_DEPTH = 20
RANDOMNESS = 0.25

# The code here is entirely unused for now except for the `main` (for testing)

def index_2d(x, y, size_x):
	return y * size_x + x

def analyze_white(board):
	global ANALYSIS_DEPTH

	with engine.SimpleEngine.popen_uci('/content/stockfish') as sf:
		result = sf.analyze(board, engine.Limit(depth=ANALYSIS_DEPTH))
		score = result['score'].white().score() - result['score'].black().score()
		return score

def analyze_black(board):
	global ANALYSIS_DEPTH

	with engine.SimpleEngine.popen_uci('/content/stockfish') as sf:
		result = sf.analyze(board, engine.Limit(depth=ANALYSIS_DEPTH))
		score = result['score'].black().score() - result['score'].white().score()
		return score

def split_per_piece(board):
	split_board = numpy.zeros((14, 8, 8), dtype=numpy.int8)

	for piece in chess.PIECE_TYPES:
		for idx, cell in enumerate(zip(board.pieces(piece, chess.WHITE), board.pieces(piece, chess.BLACK))):
			rank = chess.square_rank(cell)
			file = chess.square_rank(file)

			split_board[piece_idx][file][rank] = 1
			piece_idx += 1

def reform_state(state):
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

def main():
	'''
		Temporary testing.
	'''
	env = gym.make('ChessVsSelf-v0', render_mode='image-cl')
	model = load_model('ChessDNN.h5')
	agent = DNNChessAgent(model, env)

	state = env.reset()

	terminated = False
	last_turn_2 = -1
	last_turn_1 = -1

	dnn_dir = os.path.join('episodes', 'dnn')

	if not os.path.exists(dnn_dir):
		os.makedirs(dnn_dir)
	else:
		for filename in os.listdir(dnn_dir):
			file = os.path.join(dnn_dir, filename)

			if os.path.isfile(file):
				os.unlink(file)
			else:
				shutil.rmtree(file)

	while not bool(terminated):
		state = reform_state(state)

		move = agent.get_move(state)
		
		next_state = None

		# 	except: pass
		next_state, reward, terminated, truncated, info = env.step(move)

		print(terminated)
		print(info)
		env.render()
		plot_model(model, show_shapes=True, show_layer_activations=True)
		state = next_state

		last_turn_2 = last_turn_1
		last_turn_1 = len(info['board'].move_stack)
		
	env.render()

if __name__ == '__main__':
	main()