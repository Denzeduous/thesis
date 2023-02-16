import math
import time
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

SAMPLING = 10
EPISODES = 100
EPISODE_TIME = deque(maxlen = EPISODES // (SAMPLING // 100 + 1))

group_train_start_time = time.time()

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
	env = lib.gymchess.ChessEnv(render_mode='image-cl', folder=f'episodes_{name}', render_sampling=SAMPLING)
	#env.episode = 2_500

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	#model = build_dnn_dynamic(states, actions, optimizer, loss)
	model = load_model(f'{name}.h5')

	print(model.count_params())
	print(model.summary())

	chess_agent = ChessAgent(model, env, states)
	agent = build_agent(model, chess_agent, states, actions, env, name)
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

		agent.replay(150)

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
			total_estimated_time = avg_time * EPISODES * 6 - (time.time() - group_train_start_time)
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
	plt.savefig(f'{name}_{total_threefolds}_{total_checkmates}.png')

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':
	tests = [
		# (SGD(learning_rate=0.0001), 'huber_loss', 'ChessDNN_SGD_Huber'),
		# (SGD(learning_rate=0.0001), 'mse',        'ChessDNN_SGD_MSE'),
		# (SGD(learning_rate=0.0001), 'mae',        'ChessDNN_SGD_MAE'),
		# (SGD(learning_rate=0.0001), 'log_cosh',   'ChessDNN_SGD_LOGCOSH'),
		# (SGD(learning_rate=0.0001), 'poisson',    'ChessDNN_SGD_POISSON'),

		# (Adam(learning_rate=0.0001), 'huber_loss', 'ChessDNN_ADAM_Huber'),
		(Adam(learning_rate=0.0001), 'mse',        'ChessDNN'),
		# (Adam(learning_rate=0.0001), 'mae',        'ChessDNN_ADAM_MAE'),
		# (Adam(learning_rate=0.0001), 'log_cosh',   'ChessDNN_ADAM_LOGCOSH'),
		# (Adam(learning_rate=0.0001), 'poisson',    'ChessDNN_ADAM_POISSON'),

		# (RMSprop(learning_rate=0.0001), 'huber_loss', 'ChessDNN_RMSPROP_Huber'),
		# (RMSprop(learning_rate=0.0001), 'mse',        'ChessDNN_RMSPROP_MSE'),
		# (RMSprop(learning_rate=0.0001), 'mae',        'ChessDNN_RMSPROP_MAE'),
		# (RMSprop(learning_rate=0.0001), 'log_cosh',   'ChessDNN_RMSPROP_LOGCOSH'),
		# (RMSprop(learning_rate=0.0001), 'poisson',    'ChessDNN_RMSPROP_POISSON'),
	]

	for test in tests:
		train_dnn(test[0], test[1], test[2])