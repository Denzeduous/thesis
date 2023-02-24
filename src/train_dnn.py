import math
import time
import itertools
import chess
import gymnasium as gym
import lib.gymchess
import numpy as np
from math import floor
from lib.dnn import ChessAgent, QLearnAgent
from matplotlib import pyplot as plt
from collections import deque
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras import Model
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, Input, Lambda, LeakyReLU, Softmax
from keras.layers.merge import concatenate
from keras.losses import SparseCategoricalCrossentropy

import pandas as pd
from pandas.core.common import flatten
import sys

SAMPLING = 100
EPISODES = 5_000
EPISODE_TIME = deque(maxlen = EPISODES // max(SAMPLING // 100, 1))

def build_dnn_old(states, actions):
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

def build_dnn_goal(states, actions):
	state = Input(shape=(1, states))

	layer_1 = Dense(1024, activation='relu')(state)
	drop_1  = Dropout(0.25)(layer_1)
	layer_2 = Dense(1024, activation='relu')(drop_1)
	drop_2  = Dropout(0.25)(layer_2)
	layer_3 = Dense(1024, activation='relu')(drop_2)
	drop_3  = Dropout(0.25)(layer_3)
	layer_4 = Dense(1024, activation='relu')(drop_3)
	drop_4  = Dropout(0.25)(layer_4)
	layer_5 = Dense(1024, activation='relu')(drop_4)
	drop_5  = Dropout(0.25)(layer_5)
	layer_6 = Dense(1024, activation='relu')(drop_5)
	drop_6  = Dropout(0.25)(layer_6)

	from_layer = Dense(64, activation='linear')(layer_6)
	to_layer   = Dense(64, activation='linear')(layer_6)
	pro_layer  = Dense(4,  activation='linear')(layer_6)

	from_probability = Softmax()(from_layer)
	to_probability   = Softmax()(to_layer)
	pro_probability  = Softmax()(pro_layer)

	action_probabilities = concatenate([from_probability, to_probability, pro_probability])

	model = Model(inputs=state, outputs=action_probabilities)
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return model

def build_dnn_split(states, actions):
	state = Input(shape=(1, states))

	layer_1 = Dense(65, activation='relu')(state)
	layer_2 = Dense(65, activation='relu')(layer_1)
	layer_3 = Dense(65, activation='relu')(layer_2)
	output = Dense(actions, activation='linear')(layer_3)

	action_probabilities = concatenate([from_probability, to_probability, pro_probability])

	optimizer = Adam(learning_rate=0.001, clipnorm=1.0, clipvalue=0.5)

	model = Model(inputs=state, outputs=action_probabilities)
	model.compile(optimizer=optimizer,
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return model

def build_dnn(states, actions):
	state = Input(shape=(1, states))

	layer_1 = Dense(1024, activation='relu')(state)
	layer_2 = Dense(1024, activation='relu')(layer_1)
	layer_3 = Dense(1024, activation='relu')(layer_2)
	layer_4 = Dense(1024, activation='relu')(layer_3)
	output  = Dense(actions, activation='linear')(layer_4)

	optimizer = SGD(learning_rate=0.0001)#, clipnorm=1, clipvalue=0.5)#, decay=0.01)

	model = Model(inputs=state, outputs=output)
	model.compile(optimizer=optimizer,
	              loss='mse',
	              metrics=['accuracy'])

	return model

def build_dnn_dynamic(states, actions, optimizer, loss):
	state = Input(shape=(1, states))

	layer_1 = Dense(1024, activation='relu')(state)
	layer_2 = Dense(1024, activation='relu')(layer_1)
	layer_3 = Dense(1024, activation='relu')(layer_2)
	layer_4 = Dense(1024, activation='relu')(layer_3)
	output  = Dense(actions, activation='linear')(layer_4)

	model = Model(inputs=state, outputs=output)
	model.compile(optimizer=optimizer,
	              loss=loss,
	              metrics=['accuracy'])

	return model

def build_agent(model, chess_agent, states, actions, env, name = 'ChessDNN'):
	return QLearnAgent(model, chess_agent, name, env, states, EPISODES)

def train_dnn(optimizer, loss, name):
	env = lib.gymchess.ChessEnv(render_mode='image', folder=f'episodes_{name}', render_sampling=SAMPLING)
	env.episode = 700

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnn_dynamic(states, actions, optimizer, loss)
	#model = load_model(f'{name}.h5')

	print(model.count_params())
	print(model.summary())

	chess_agent = ChessAgent(model, env, states)
	agent = build_agent(model, chess_agent, states, actions, env, name)

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

			next_state = None
			reward = None
			terminated = None

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
					last_memory = (last_memory[0], last_memory[1], -reward / 10, last_memory[3], last_memory[4])
					del agent.memory[-1]
					agent.remember(*last_memory)

				else:
					mem = agent.memory[-1]
					agent.memory[-1] = (mem[0], mem[1], mem[2] - reward / 10, mem[3], mem[4])

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

	plt.plot(agent.loss)
	plt.title('Model Performance')
	plt.ylabel('Loss')
	plt.xlabel('Episode')
	plt.legend(['loss'], loc='upper left')
	plt.savefig(f'{name}.png')
	plt.clf()
	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

def test_dnn_types(tests):
	start = time.time()
	lineup = itertools.permutations(tests, 2)
	delta = time.time() - start

	for x in lineup:
		name1 = x[0]
		name2 = x[1]

		env = lib.gymchess.ChessEnv(render_mode='image-cl', folder=f'vs/dnn/{name1}_vs_{name2}', render_sampling=SAMPLING)
		#env.episode = 2_500

		states = env.observation_space['board'].n + env.observation_space['player'].n
		actions = env.action_space.n

		#model = build_dnn_dynamic(states, actions, optimizer, loss)
		model1 = load_model(f'{name1}.h5')
		model2 = load_model(f'{name2}.h5')

		chess_agent1 = ChessAgent(model1, env, states)
		chess_agent2 = ChessAgent(model2, env, states)

		white = True
		reward_white = []
		reward_black = []

		state = env.reset()

		terminated = False

		while not terminated:
			action, pred_from, pred_to = None, None, None

			if white:
				action, pred_from, pred_to = chess_agent1.get_move(state)
			else:
				action, pred_from, pred_to = chess_agent2.get_move(state)

			next_state = None
			reward = None
			terminated = None

			next_state, reward, terminated, truncated, info = env.step(action, pred_from=pred_from, pred_to=pred_to)

			env.render()

			reward = 1 if reward == 0 else reward

			if white:
				reward_white.append(reward)
			else:
				reward_black.append(reward)

			state = next_state

			if not terminated:
				white = not white

		env.render()

		plt.plot(reward_white)
		plt.plot(reward_black)
		plt.title('Model Performance')
		plt.ylabel('Rewards')
		plt.xlabel('Epoch')
		plt.legend(['white', 'black'], loc='upper left')
		plt.savefig(f'vs/dnn/{name1}_{sum(reward_white)}_{name2}_{sum(reward_black)}.png')
		plt.clf()
		nm_1, nm_2 = 'white', 'black'

		print(f'Ended with {terminated} with winner being {nm_1 if white else nm_2} ({name1 if white else name2})')

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
	
	#if not isinstance(state['ownership'], np.ndarray):
		#state['ownership'] = np.concatenate(state['ownership'])

	if not isinstance(state['player'], int):
		state['player'] = 0 if state['player'] == 'White' else 1
	
	return np.array(list(flatten(state.values()))).reshape(1, states)

def train_dnn_db(optimizer, loss, name):
	env = lib.gymchess.ChessEnv(render_mode=None)
	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnn_dynamic(states, actions, optimizer, loss)

	df = pd.read_csv('games.csv')

	model_states = []
	model_targets = []

	size = len(df.index)

	for idx, row in df.iterrows():
		winner = row.winner == 'white' # False is Black
		player = True # White, False is Black

		state = env.reset()

		for move_san in row.moves.split(' '):
			move = env.board.parse_san(move_san)

			from_square = move.from_square
			to_square = move.to_square
			promotion = move.promotion

			next_state, reward, _, _, _ = env.step(move)

			if player == winner:
				target = np.zeros(actions)

				target[from_square] = reward
				target[to_square + 64] = reward

				if promotion != None:
					target[promotion + 128 - 2] = reward

				model_targets.append(target)
				model_states.append(reform_state(state, states))

			player = not player
			state = next_state

		progress_bar(idx / size * 10)

	model_states = np.array(model_states)
	model_targets = np.array(model_targets)

	history = model.fit(model_states, model_targets, epochs=1)
	loss = history.history['loss']

	plt.plot(loss)
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['loss'], loc='upper left')
	plt.savefig(f'ChessDNN_Non-RL.png')
	plt.clf()

	model.save(f'{name}.h5')
	print('\nDone\n')

if __name__ == '__main__':
	optimizer = Adam(learning_rate=0.001)
	loss = 'mse'
	name = 'ChessDNN'

	train_dnn_db(optimizer, loss, name)
	train_dnn(optimizer, loss, name)

	#test_dnn_types(['ChessDNN_nonrl', 'ChessDNN_nonrl_001', 'ChessDNN_nonrl_0001'])