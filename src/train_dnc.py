import math
import time
import chess
import itertools
import gymnasium as gym
import lib.gymchess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop
from torch.nn import HuberLoss, MSELoss, L1Loss, GaussianNLLLoss, PoissonNLLLoss
from math import floor
from lib.dnc import ChessAgent, QLearnAgent, SequentialDNC
from matplotlib import pyplot as plt
from collections import deque

import pandas as pd
from pandas.core.common import flatten
import sys

SAMPLING = 100
EPISODES = 5_000
EPISODE_TIME = deque(maxlen = EPISODES // max(SAMPLING // 100, 1))

def build_dnc(states, actions):
	return SequentialDNC(states, actions, training=True)

def build_agent(model, chess_agent, states, actions, env, name, loss, optimizer):
	return QLearnAgent(model, chess_agent, name, env, states, EPISODES, loss, optimizer)

def train_dnc(optimizer, loss, name):
	env = lib.gymchess.ChessEnv(render_mode='image', folder=f'episodes_{name}', render_sampling=SAMPLING)

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnc(states, actions)
	#model = torch.load(f'{name}.pth.tar')

	print(f"Model structure: {model}\n\n")

	chess_agent = ChessAgent(model, env, states)
	agent = build_agent(model, chess_agent, states, actions, env, name, loss, optimizer)

	train_start_time = time.time()

	total_threefolds = 0
	total_checkmates = 0
	threefolds = 0
	checkmates = 0

	for i in range(EPISODES if name != 'ChessDNC_ADAM_MSE' else 2_100):
		start_time = time.time()

		state = env.reset()

		terminated = False
		last_memory = None

		sample = (i + 1) % SAMPLING == 0 or i == 0

		while not terminated:
			action, pred_from, pred_to = agent.step(env, state)
			next_state, reward, terminated, truncated, info = env.step(action, pred_from=pred_from, pred_to=pred_to)

			env.render()

			if bool(terminated):
				if terminated.termination == chess.Termination.THREEFOLD_REPETITION:
					total_threefolds += 1
					threefolds += 1
				elif terminated.termination == chess.Termination.CHECKMATE:
					total_checkmates += 1
					checkmates += 1

			if reward > 0:
				if last_memory != None:
					last_memory = (last_memory[0], last_memory[1], -reward, last_memory[3], last_memory[4])
					del agent.memory[-1]
					agent.remember(*last_memory)

				else:
					mem = agent.memory[-1]
					agent.memory[-1] = (mem[0], mem[1], mem[2] - reward, mem[3], mem[4], mem[5], mem[6])

				last_memory = None

			else:
				last_memory = (state, action, reward, next_state, terminated)

			agent.remember(state, action, reward, next_state, terminated)
			
			state = next_state

		agent.replay(100)

		# Calculate time remaining for training.
		end_time = time.time()
		delta_time = end_time - start_time
		delta_min = round(delta_time / 60)
		delta_sec = round(delta_time % 60)
		EPISODE_TIME.append(delta_time)

		if sample:
			avg_time = np.average(EPISODE_TIME)
			elapsed_time = time.time() - train_start_time
			elapsed_min = round(elapsed_time / 60)
			elapsed_sec = round(elapsed_time % 60)
			estimated_time = avg_time * EPISODES - elapsed_time
			estimated_min = round(estimated_time / 60)
			estimated_sec = round(estimated_time % 60)
			estimate_samp = estimated_time / ((EPISODES - i) / SAMPLING)
			sample_min = round(estimate_samp / 60)
			sample_sec = round(estimate_samp % 60)

			print(f'Episode {i + 1} finished in {delta_min} minutes and {delta_sec} seconds because of {terminated.termination}.')
			print(f'There were a total of {threefolds} threefold repetitions and {checkmates} checkmates.')
			print(f'Exploration rate is {int(agent.epsilon * 100)}% ({agent.epsilon}).')
			print(f'Elapsed time: {elapsed_min} minutes and {elapsed_sec} seconds.')
			print(f'Estimated time until next sample: {sample_min} minutes and {sample_sec} seconds.')
			print(f'Estimated time remaining: {estimated_min} minutes and {estimated_sec} seconds.\n')

			threefolds = 0
			checkmates = 0
			agent.save_model()

			plt.plot(agent.loss)
			plt.title('Model Performance')
			plt.ylabel('Loss')
			plt.xlabel('Episode')
			plt.legend(['loss'], loc='upper left')
			plt.savefig(f'{name}.png')
			plt.clf()

		env.render()

	agent.save_model()

	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.title('Model Performance')
	plt.ylabel('Loss (Normalized)')
	plt.xlabel('Epoch')
	plt.legend(['loss'], loc='upper left')
	plt.savefig(f'{name}_{total_threefolds}_{total_checkmates}.png')
	plt.clf()

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

def progress_bar(ratio, width=20, fill_char='='):
	width -= len('[] 0%')
	sys.stdout.write('\r')
	sys.stdout.write('[%s] %3d%%' % (fill_char * int(ratio * width), ratio * 10))
	sys.stdout.flush()

def reform_state(state, states):
	'''
		Reforms the state into a flattened 1D array.
		
		The `isinstance` calls within are to make sure
		that it hasn't been called before on the same state.

		Unfortunately, this has to be copied from the QLearn.
	'''
	if isinstance(state, np.ndarray):
		return state.reshape(1, states)

	if isinstance(state, tuple):
		state = state[0] # No idea why this happens tbh

	if not isinstance(state['board'], np.ndarray):
		state['board'] = np.concatenate(state['board'])

	if not isinstance(state['player'], int):
		state['player'] = 0 if state['player'] == 'White' else 1
	
	return np.array(list(flatten(state.values()))).reshape(1, states)

def train_dnc_db(optimizer, loss, name):
	env = lib.gymchess.ChessEnv(render_mode=None)
	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnc(states, actions)

	optimizer = optimizer(model.parameters(), lr=0.001)

	df = pd.read_csv('games.csv')

	size = len(df.index)
	losses = []
	model.train()

	for idx, row in df.iterrows():
		winner = row.winner == 'white' # False is Black
		player = True # White, False is Black

		state = env.reset()

		with torch.no_grad():
			model.reset()

		for move_san in row.moves.split(' '):
			move = env.board.parse_san(move_san)

			from_square = move.from_square
			to_square = move.to_square
			promotion = move.promotion

			next_state, reward, _, _, _ = env.step(move)

			if player == winner:
				# Get target output (what move was actually made
				# in the format of the neural network with rewards)
				target = np.zeros(actions, dtype=np.float32)

				target[from_square] = reward
				target[to_square + 64] = reward

				if promotion != None:
					target[promotion + 128 - 2] = reward

				# Get prediction
				output = model(torch.tensor(reform_state(state, states)))

				# Backpropagation
				model_loss = loss(output, torch.tensor(target.reshape(1, 132)))
				optimizer.zero_grad()
				model_loss.backward()

				# Optimization
				optimizer.step()

				# Save loss for graph
				with torch.no_grad():
					losses.append(model_loss.detach().numpy())

			player = not player
			state = next_state

		progress_bar(idx / size * 10)

	plt.plot(losses)
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['loss'], loc='upper left')
	plt.savefig(f'ChessDNC_Non-RL.png')
	plt.clf()

	with open(f'{name}.pth.tar', 'wb+') as file:
		torch.save(model, file)

	print('\nDone\n')

if __name__ == '__main__':
	optimizer = Adam
	loss = MSELoss()
	name = 'ChessDNC'

	#train_dnc_db(optimizer, loss, name)
	train_dnc(optimizer, loss, name)