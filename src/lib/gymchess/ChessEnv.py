import os
import shutil
import gymnasium as gym
import chess
import chess.svg
import chess.engine
import numpy as np
import gymnasium as gym
from lib.util import prediction_to_color
from gymnasium import spaces

class ChessEnv(gym.Env):
	metadata = {'render_modes': ['image', 'image-cl', 'fen']}

	def __init__(self, render_mode: str = None, render_sampling: int = 1, reward_type: str = 'Simple', folder='episodes', analysis_depth: int = 10):
		# 8x8 board with the player
		self.observation_space = spaces.Dict({
			'board': spaces.Discrete(64),
			#'ownership': spaces.Discrete(64),
			'player': spaces.Discrete(1),
		})

		self.episode = 0
		self.folder = folder

		# Every board position * 2 + promotions
		self.action_space = spaces.Discrete(64 * 2 + 4)
		
		self.render_mode = render_mode
		self.render_sampling = render_sampling
		self.reward_type = reward_type
		self.analysis_depth = analysis_depth
		self.colors = None

		rewards = ['Simple', 'Stockfish', None]

		if reward_type not in rewards:
			rewards_str = ', '.join(rewards)
			raise NotImplementedError(f'Invalid reward type "{reward_type}". Valid reward types are: {rewards_str}.')

		self.board_data = np.vectorize(lambda x: int(x.piece_type) * (-1 if x.color != self.board.turn else 1) if x != None else 0)
		self.player_data = np.vectorize(lambda x: int(x.color) + 1 if x != None else 0)
		#self.engine = chess.engine.SimpleEngine.popen_uci('../stockfish.exe')

		self.reset()

		if render_mode == 'image' or render_mode == 'image-cl':
			if not os.path.exists(self.folder):
				os.makedirs(self.folder)

		if render_mode == 'image-cl':
			if os.path.exists(self.folder):
				for filename in os.listdir(self.folder):
					file = os.path.join(self.folder, filename)

					if os.path.isfile(file):
						os.unlink(file)
					else:
						shutil.rmtree(file)

	@property
	def possible_actions(self):
		return np.array([move for move in self.board.legal_moves])

	def reset(self, seed=None, board: chess.Board = None):
		super().reset(seed=seed)

		self.seed = seed
		self.last_render = None

		if hasattr(self, 'board'):
			self.episode += 1

		if board == None:
			self.board = chess.Board()
		else:
			self.board = board

		# This follows the indexing at
		# https://python-chess.readthedocs.io/en/latest/core.html#colors
		# and
		# https://python-chess.readthedocs.io/en/latest/core.html#piece-types
		self.pieces = [
			[
				8, 2, 2, 2, 1, 1,
			],
			[
				8, 2, 2, 2, 1, 1,
			],
		]

		self.player = 'White'

		return self._get_obs(), self._get_info()

	def step(self, move: chess.Move, pred_from = None, pred_to = None):
		player = self.board.turn

		if move.promotion != None:
			self.pieces[player][chess.PAWN] -= 1
			self.pieces[player][move.promotion] += 1

		self.board.push(move)

		pieces = [
			[
				len(square) for square in [
					self.board.pieces(chess.PAWN,   chess.WHITE),
					self.board.pieces(chess.KNIGHT, chess.WHITE),
					self.board.pieces(chess.BISHOP, chess.WHITE),
					self.board.pieces(chess.ROOK,   chess.WHITE),
					self.board.pieces(chess.QUEEN,  chess.WHITE),
					self.board.pieces(chess.KING,   chess.WHITE),
				]
			],
			[
				len(square) for square in [
					self.board.pieces(chess.PAWN,   chess.BLACK),
					self.board.pieces(chess.KNIGHT, chess.BLACK),
					self.board.pieces(chess.BISHOP, chess.BLACK),
					self.board.pieces(chess.ROOK,   chess.BLACK),
					self.board.pieces(chess.QUEEN,  chess.BLACK),
					self.board.pieces(chess.KING,   chess.BLACK),
				]
			]
		]

		reward = 0

		if self.reward_type == 'Simple':
			opponent_pieces = [
				pieces[not player][piece] * 10 for piece in range(len(pieces[not player]))
				if pieces[not player][piece] < self.pieces[not player][piece]
			]

			reward = int(sum(opponent_pieces) * 1.5)

			if self.board.is_checkmate():
				reward += 10

			elif self.board.is_check():
				reward += 3

			elif self.board.outcome() != None:
				reward -= 30

			if move in self.board.move_stack:
				reward -= 30

			# for piece in range(len(opponent_pieces)):
			# 	if opponent_pieces[piece] < self.pieces[not player][piece]:
			# 		if piece == chess.KING:
			# 			reward = 100
			# 		else:
			# 			reward = piece * 10

			# 		break

		self.pieces = pieces

		if self.reward_type == 'Stockfish':
			if self.player == 'White':
				reward = self._analyze_white(self.board)
			else:
				reward = self._analyze_black(self.board)

		if self.player == 'White':
			self.player = 'Black'
		else:
			self.player = 'White'
		
		if pred_from is not None and pred_to is not None:
			self.colors = prediction_to_color(pred_from, pred_to)
		else:
			self.colors = None

		return self._get_obs(), reward, self.board.outcome(claim_draw = True), False, self._get_info()

	def render(self):
		if (self.render_mode == 'image' or self.render_mode == 'image-cl') and (self.episode % self.render_sampling == 0 or self.episode == 1):
			episode_dir = os.path.join(self.folder, f'Episode_{self.episode}')
			file_path = os.path.join(episode_dir, f'move_{len(self.board.move_stack):03d}.svg')

			if not os.path.exists(episode_dir):
				os.makedirs(episode_dir)

			with open(file_path, 'w+') as f:
				if len(self.board.move_stack) == 0:
					f.write(chess.svg.board(self.board))

				else:
					move = self.board.peek()

					if self.last_render != move:
						if self.colors is not None:
							f.write(chess.svg.board(
								self.board,
								fill=self.colors,
								arrows=[chess.svg.Arrow(move.from_square, move.to_square, color='#ff2222')],
							))
						else:
							f.write(chess.svg.board(
								self.board,
								arrows=[chess.svg.Arrow(move.from_square, move.to_square, color='#ff2222')],
							))
					else:
						f.write(chess.svg.board(self.board))

					self.last_render = move

		elif self.render_mode == 'fen':
			return self.board.board_fen()

	def _get_obs(self):
		board = np.array([[self.board.piece_at(x + y * 8) for x in range(7, -1, -1)] for y in range(7, -1, -1)]).reshape(64)

		values = self.board_data(board)

		return {
			'board': values,
			'player': self.player,
		}

	def _get_info(self):
		return {
			'board': self.board,
		}

	def _analyze_white(self, board):
		result = self.engine.analyse(board, chess.engine.Limit(depth=self.analysis_depth))

		white = result['score'].white().score()
		black = result['score'].black().score()
		
		if white == None or black == None:
			return 0

		return white - black

	def _analyze_black(self, board):
		result = self.engine.analyse(board, chess.engine.Limit(depth=self.analysis_depth))

		white = result['score'].white().score()
		black = result['score'].black().score()
		
		if white == None or black == None:
			return 0

		return black - white