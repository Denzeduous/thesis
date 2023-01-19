import gymnasium as gym
import lib.gymchess as gym_chess
import chess
import chess.engine
import numpy as np
import random as rand
from math import floor
from keras.models import load_model
from pandas.core.common import flatten

ANALYSIS_DEPTH = 20

# The code here is entirely unused for now except for the `main` (for testing)

def index_2d(x, y, size_x):
	return y * size_x + x

def analyze(board, color: str):
	if color == 'WHITE':
		return analyze_white(board)
	return analyze_black(board)

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
	if isinstance(state, np.ndarray):
		return state.reshape(1, 1, 71)

	if not isinstance(state['board'], np.ndarray):
		state['board'] = np.concatenate(state['board'])
	
	if not isinstance(state['current_player'], int):
		state['current_player'] = 0 if state['current_player'] == 'WHITE' else 1
	
	return np.array(list(flatten(state.values()))).reshape(1, 1, 71)