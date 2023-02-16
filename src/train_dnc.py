import math
import time
import chess
import gymnasium as gym
import lib.gymchess
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD, RMSprop
from torch.nn import HuberLoss, MSELoss, L1Loss, GaussianNLLLoss, PoissonNLLLoss
from math import floor
from lib.dnc import ChessAgent, QLearnAgent, SequentialDNC
from matplotlib import pyplot as plt

SAMPLING = 10
EPISODES = 100
EPISODE_TIME = np.zeros(EPISODES)

group_train_start_time = time.time()

def build_dnc(states, actions):
	return SequentialDNC(states, actions, training=True)

def build_agent(model, chess_agent, states, actions, env, name, loss, optimizer):
	return QLearnAgent(model, chess_agent, name, env, states, EPISODES, loss, optimizer)

def train_dnc(optimizer, loss, name):
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder=f'episodes_{name}', render_sampling=SAMPLING)
	#env.episode = 5_000

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	#model = build_dnc(states, actions)
	model = torch.load(f'{name}.pth.tar')

	print(f"Model structure: {model}\n\n")

	chess_agent = ChessAgent(model, env, states)
	agent = build_agent(model, chess_agent, states, actions, env, name, loss, optimizer)
	agent.epsilon = 0

	train_start_time = time.time()

	total_threefolds = 0
	total_checkmates = 0
	threefolds = 0
	checkmates = 0

	for i in range(EPISODES):
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

		agent.replay(150)

		# Calculate time remaining for training.
		end_time = time.time()
		delta_time = end_time - start_time
		delta_min = round(delta_time / 60)
		delta_sec = round(delta_time % 60)
		EPISODE_TIME[i] = delta_time

		if sample:
			avg_time = np.average(EPISODE_TIME)
			elapsed_time = time.time() - train_start_time
			elapsed_min = round(elapsed_time / 60)
			elapsed_sec = round(elapsed_time % 60)
			total_estimated_time = avg_time * EPISODES * 5 - (time.time() - group_train_start_time)
			total_estimated_min = round(total_estimated_time / 60)
			total_estimated_sec = round(total_estimated_time % 60)
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
			print(f'Estimated time remaining: {estimated_min} minutes and {estimated_sec} seconds.')
			#print(f'Total estimated time until testing is complete: {total_estimated_min} minutes and {total_estimated_sec} seconds.\n')
			threefolds = 0
			checkmates = 0
			agent.save_model()

		env.render()

	agent.save_model()

	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.title('Model Performance')
	plt.ylabel('Loss (Normalized)')
	plt.xlabel('Epoch')
	plt.legend(['loss'], loc='upper left')
	plt.savefig(f'{name}-2_{total_threefolds}_{total_checkmates}.png')

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':
	tests = [
		#(SGD, HuberLoss(),       'ChessDNC_SGD_Huber'),
		# (SGD, MSELoss(),         'ChessDNC_SGD_MSE'),
		#(SGD, L1Loss(),          'ChessDNC_SGD_MAE'),

		# (Adam, HuberLoss(),       'ChessDNC_ADAM_Huber'),
		(Adam, MSELoss(),         'ChessDNC'),
		# (Adam, L1Loss(),          'ChessDNC_ADAM_MAE'),

		# (RMSprop, HuberLoss(),       'ChessDNC_RMSPROP_Huber'),
		#(RMSprop, MSELoss(),         'ChessDNC_RMSPROP_MSE'),
		# (RMSprop, L1Loss(),          'ChessDNC_RMSPROP_MAE'),
	]
	
	for test in tests:
		train_dnc(test[0], test[1], test[2])