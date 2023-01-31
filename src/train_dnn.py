import math
import time
import chess
import gymnasium as gym
import lib.gymchess
import lib.chessutil
import numpy as np
from math import floor
from lib.dnn import ChessAgent, QLearnAgent
from matplotlib import pyplot as plt
from collections import deque
from keras import Model
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, Input, Lambda, LeakyReLU, Softmax
from keras.layers.merge import concatenate
from keras.optimizers import adam_v2 as Adam
from keras.losses import SparseCategoricalCrossentropy

SAMPLING = 500
EPISODES = 5_000
EPISODE_TIME = deque(maxlen = EPISODES // SAMPLING)

def build_dnn_old(states, actions):
	print(states, actions)
	model = Sequential()
	model.add(Flatten(input_shape=(1, 1, states)))

	model.add(Dense(64,  activation='tanh'))
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(132))
	model.add(LeakyReLU(alpha=0.05))

	model.add(Dense(actions, activation='softmax'))
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	
	return model

def build_dnn(states, actions):
	state = Input(shape=(1, states))

	from_square_l1   = Dense(65, activation='tanh')(state)
	from_square_l2   = Dense(65, activation='tanh')(from_square_l1)
	from_square_l3   = Dense(64)(from_square_l2)
	from_leaky_relu  = LeakyReLU(alpha=0.05)(from_square_l3)
	from_probability = Softmax()(from_square_l3)

	to_square_l1   = Dense(65, activation='tanh')(state)
	to_square_l2   = Dense(65, activation='tanh')(to_square_l1)
	to_square_l3   = Dense(64)(to_square_l2)
	to_leaky_relu  = LeakyReLU(alpha=0.05)(to_square_l3)
	to_probability = Softmax()(to_square_l3)

	pro_square_l1   = Dense(65, activation='tanh')(state)
	pro_square_l2   = Dense(65, activation='tanh')(pro_square_l1)
	pro_square_l3   = Dense(4)(pro_square_l2)
	pro_leaky_relu  = LeakyReLU(alpha=0.05)(pro_square_l3)
	pro_probability = Softmax()(pro_square_l3)

	action_probabilities = concatenate([from_probability, to_probability, pro_probability])

	model = Model(inputs=state, outputs=action_probabilities)
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return model

def build_agent(model, chess_agent, states, actions, env):
	return QLearnAgent(model, chess_agent, 'ChessDNN', env, states, EPISODES)

def train_dnn():
	env = lib.gymchess.ChessEnv(render_mode='image', render_sampling=SAMPLING)
	env.episode = 5_000

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	#model = build_dnn(states, actions)
	model = load_model('ChessDNN.h5')

	print(model.count_params())
	print(model.summary())

	chess_agent = ChessAgent(model, env)
	agent = build_agent(model, chess_agent, states, actions, env)

	train_start_time = time.time()

	for i in range(EPISODES):
		start_time = time.time()

		state = env.reset()
		terminated = False

		sample = (i + 1) % SAMPLING == 0 or i == 0

		while not terminated:
			action, pred_from, pred_to = agent.step(env, state)

			next_state = None
			reward = None
			terminated = None

			next_state, reward, terminated, truncated, info = env.step(action, pred_from=pred_from, pred_to=pred_to)

			env.render()

			if reward >= 0 and not bool(terminated):
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
			agent.save_model()

		env.render()

	agent.save_model()

	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.title('Model Performance')
	plt.ylabel('Loss (Normalized)')
	plt.xlabel('Epoch')
	plt.legend(['loss'], loc='upper left')
	plt.show()

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':
	train_dnn()