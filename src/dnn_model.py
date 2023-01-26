import math
import time
import chess
import gymnasium as gym
import lib.gymchess
import lib.chessutil
import numpy as np
from math import floor
from lib.dnn import DNNChessAgent
from lib.dnnqlearn import DNNQLearnAgent
from matplotlib import pyplot as plt
from collections import deque
from keras import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Softmax
from keras.optimizers import adam_v2 as Adam
from keras.losses import SparseCategoricalCrossentropy

SAMPLING = 500
EPISODES = 5_000
EPISODE_TIME = deque(maxlen = EPISODES // SAMPLING)
REWARDS = [[] for x in range(EPISODES)]

def build_dnn(states, actions):
	print(states, actions)
	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(Dense(64,  activation='tanh'))
	model.add(Dense(actions))
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	
	return model

def build_agent(model, chess_agent, states, actions, env):
	return DNNQLearnAgent(model, chess_agent, 'ChessDNN', env, states, EPISODES)

def train_dnn():
	global REWARDS
	env = gym.make('ChessVsSelf-v0', render_mode='image', render_sampling=SAMPLING)
	env.episode = 1000

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	#model = build_dnn(states, actions)
	model = load_model('ChessDNN.h5')

	print(model.count_params())
	print(model.summary())

	chess_agent = DNNChessAgent(model, env)
	agent = build_agent(model, chess_agent, states, actions, env)

	train_start_time = time.time()

	for i in range(EPISODES):
		start_time = time.time()

		state = env.reset()
		terminated = False

		sample = (i + 1) % SAMPLING == 0 or i == 0

		while not terminated:
			action = agent.step(env, state)

			next_state = None
			reward = None
			terminated = None

			next_state, reward, terminated, truncated, info = env.step(action)

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
		EPISODE_TIME.append(delta_time)

		if sample:
			avg_time = np.average(EPISODE_TIME)
			elapsed_time = time.time() - train_start_time
			elapsed_min = round(elapsed_time / 60)
			elapsed_sec = round(elapsed_time % 60)
			estimated_time = avg_time * EPISODES - elapsed_time
			estimated_min = round(estimated_time / 60)
			estimated_sec = round(estimated_time % 60)

			print(f'Episode {i + 1} finished in {delta_min} minutes and {delta_sec} seconds because of {terminated.termination}.')
			print(f'Elapsed time: {elapsed_min} minutes and {elapsed_sec} seconds.')
			print(f'Estimated time remaining: {estimated_min} minutes and {estimated_sec} seconds.\n')

		env.render()

	agent.save_model()

	avg_rewards = np.array([np.mean(episode) for episode in REWARDS])

	plt.plot(avg_rewards)
	plt.plot(agent.accuracy / np.linalg.norm(agent.accuracy))
	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.title('Model Performance')
	plt.ylabel('Accuracy / Rewards (Normalized)')
	plt.xlabel('Epoch')
	plt.legend(['average rewards', 'accuracy', 'loss'], loc='upper left')
	plt.show()

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':
	train_dnn()