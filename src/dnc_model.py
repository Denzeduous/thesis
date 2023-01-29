import math
import time
import chess
from dnc import DNC
import gymnasium as gym
import lib.gymchess
import lib.chessutil
import numpy as np
from math import floor
from lib.dnc import ChessAgent, QLearnAgent
from matplotlib import pyplot as plt
from keras import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Softmax
from keras.optimizers import adam_v2 as Adam
from keras.losses import SparseCategoricalCrossentropy

SAMPLING = 1
EPISODES = 10_000
EPISODE_TIME = np.zeros(EPISODES)
REWARDS = [[] for x in range(EPISODES)]

def build_dnc(states, actions):
	return DNC(states, 128, actions, 64, 128, 1, 12)

def build_agent(model, chess_agent, states, actions, env):
	return QLearnAgent(model, chess_agent, 'ChessDNC', env, states, EPISODES)

def train_dnc():
	global REWARDS
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder='episodes_dnc', render_sampling=SAMPLING)

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnc(states, actions)

	chess_agent = ChessAgent(model, env)
	agent = build_agent(model, chess_agent, states, actions, env)

	train_start_time = time.time()

	for i in range(EPISODES):
		start_time = time.time()

		state = env.reset()

		dnc_state   = model.reset()
		dnc_state_2 = model.reset()

		terminated = False

		sample = (i + 1) % SAMPLING == 0 or i == 0

		while not terminated:
			dnc_state, dnc_state_2 = dnc_state_2, dnc_state

			action, dnc_state, pred_from, pred_to = agent.step(env, state, dnc_state)

			next_state = None
			reward = None
			terminated = None

			next_state, reward, terminated, truncated, info = env.step(action, pred_from=pred_from, pred_to=pred_to)

			env.render()

			REWARDS[i].append(reward)

			if reward >= 0:
				agent.remember(state, action, reward, next_state, terminated)

			state = next_state

		agent.replay(32)

		# Calculate time remaining for training.
		end_time = time.time()
		delta_time = end_time - start_time
		delta_min = round(delta_time / 60)
		delta_sec = round(delta_time % 60)
		EPISODE_TIME[i] = delta_time

		if sample:
			avg_time = np.average(EPISODE_TIME[:i + 1])
			elapsed_time = time.time() - train_start_time
			elapsed_min = round(elapsed_time / 60)
			elapsed_sec = round(elapsed_time % 60)
			estimated_time = avg_time * EPISODES - elapsed_time
			estimated_min = round(estimated_time / 60)
			estimated_sec = round(estimated_time % 60)

			print(f'Episode {i + 1} finished in {delta_min} minutes and {delta_sec} seconds because of {terminated.termination}.')
			print(f'Elapsed time: {elapsed_min} minutes and {elapsed_sec} seconds.')
			print(f'Estimated time remaining: {estimated_min} minutes and {estimated_sec} seconds.\n')
			agent.save_model()

		env.render()

	agent.save_model()

	avg_rewards = np.array([np.mean(episode) for episode in REWARDS])

	plt.plot(avg_rewards)
	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.title('Model Performance')
	plt.ylabel('Rewards / Loss (Normalized)')
	plt.xlabel('Epoch')
	plt.legend(['average rewards', 'accuracy', 'loss'], loc='upper left')
	plt.show()

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':
	train_dnc()