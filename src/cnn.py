import time
import gym
import gym_chess
import numpy as np
from lib.qlearn import QLearnAgent
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from keras import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.optimizers import adam_v2 as Adam
from keras.losses import SparseCategoricalCrossentropy

EPISODES = 500
EPISODE_TIME = np.zeros(EPISODES)
REWARDS = [[] for x in range(EPISODES)]

def build_cnn(states):
	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(Dense(64,  activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64,  activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64,  activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64,  activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	
	return model

def build_agent(model, states, actions):
	return QLearnAgent(model, 'ChessCNN', states, actions)

def train_cnn():
	env = gym.make('ChessVsSelf-v2', log=False)

	states = 71 # 71 possible states from a flattened env state. Has to be hard-coded, unfortunately
	actions = env.action_space.n

	model = build_cnn(states, actions)
	agent = build_agent(model, states, actions)

	train_start_time = time.time()

	for i in range(EPISODES):
		print(f'Episode {i + 1} ', end='')
		start_time = time.time()

		state = env.reset()
		done = False
		j = 0

		while not done:
			action = round(agent.step(state) * len(env.possible_actions))
			next_state, reward, done, info = env.step(action)
			
			REWARDS[i].append(reward)

			agent.remember(state, action, reward, next_state, done)
			state = next_state
			j += 1

			if j > 100: break

		agent.replay(32)

		# Calculate time remaining for training.
		end_time = time.time()
		delta_time = end_time - start_time
		delta_min = round(delta_time / 60)
		delta_sec = round(delta_time % 60)
		EPISODE_TIME[i] = delta_time
		avg_time = np.average(EPISODE_TIME[:i + 1])
		elapsed_time = time.time() - train_start_time
		elapsed_min = round(elapsed_time / 60)
		elapsed_sec = round(elapsed_time % 60)
		estimated_time = avg_time * EPISODES - elapsed_time
		estimated_min = round(estimated_time / 60)
		estimated_sec = round(estimated_time % 60)

		print(f'finished in {delta_min} minutes and {delta_sec} seconds.')
		print(f'Elapsed time: {elapsed_min} minutes and {elapsed_sec} seconds.')
		print(f'Estimated time remaining: {estimated_min} minutes and {estimated_sec} seconds.\n')

	agent.save_model()

	with open('rewards.data', 'a+') as f:
		f.write(str(REWARDS))

if __name__ == '__main__':
	train_cnn()