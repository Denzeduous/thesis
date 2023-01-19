import lib.gymchess
import numpy as np
import gymnasium as gym
import tensorflow.compat.v1 as tf
from lib.dnc import DNC_test
from lib.dnc.dnc_cell import DNCCell
from lib.dnc.controller import StatelessCell
from keras import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Softmax, RNN
from keras.optimizers import adam_v2 as Adam
from keras.losses import SparseCategoricalCrossentropy

from dnc import dnc

SAMPLING = 500
EPISODES = 7_500
EPISODE_TIME = np.zeros(EPISODES)
REWARDS = [[] for x in range(EPISODES)]

FLAGS = tf.flags.FLAGS

# Hyperparameters pulled and modified from the official DNC implementation

# Model parameters
tf.flags.DEFINE_integer('hidden_size', 64, 'Size of LSTM hidden layer.')
tf.flags.DEFINE_integer('memory_size', 16, 'The number of memory slots.')
tf.flags.DEFINE_integer('word_size', 16, 'The width of each memory slot.')
tf.flags.DEFINE_integer('num_write_heads', 1, 'Number of memory write heads.')
tf.flags.DEFINE_integer('num_read_heads', 4, 'Number of memory read heads.')
tf.flags.DEFINE_integer('clip_value', 20,
						'Maximum absolute value of controller and dnc outputs.')

# Optimizer parameters
tf.flags.DEFINE_float('max_grad_norm', 50, 'Gradient clipping norm limit.')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Optimizer learning rate.')
tf.flags.DEFINE_float('optimizer_epsilon', 1e-10,
                      'Epsilon used for RMSProp optimizer.')

# Task parameters
tf.flags.DEFINE_integer('batch_size', 16, 'Batch size for training.')
tf.flags.DEFINE_integer('num_bits', 4, 'Dimensionality of each vector to copy')
tf.flags.DEFINE_integer(
	'min_length', 1,
	'Lower limit on number of vectors in the observation pattern to copy')
tf.flags.DEFINE_integer(
	'max_length', 2,
	'Upper limit on number of vectors in the observation pattern to copy')
tf.flags.DEFINE_integer('min_repeats', 1,
                        'Lower limit on number of copy repeats.')
tf.flags.DEFINE_integer('max_repeats', 2,
                        'Upper limit on number of copy repeats.')

# Training options
tf.flags.DEFINE_integer('num_training_iterations', 100000,
                        'Number of iterations to train for.')
tf.flags.DEFINE_integer('report_interval', 100,
                        'Iterations between reports (samples, valid loss).')
tf.flags.DEFINE_string('checkpoint_dir', '.\\dnc-checkpoints',
                       'Checkpointing directory.')
tf.flags.DEFINE_integer('checkpoint_interval', -1,
                        'Checkpointing step interval.')

def build_dnc(states, actions):
	access_config = {
		'memory_size': FLAGS.memory_size,
		'word_size':   FLAGS.word_size,
		'num_reads':   FLAGS.num_read_heads,
		'num_writes':  FLAGS.num_write_heads,
	}

	controller_config = {
		'hidden_size': FLAGS.hidden_size,
	}

	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(RNN(DNCCell(StatelessCell('linear', features=8), memory_size=256, word_size=8, num_reads=1, num_writes=1)))
	model.add(BatchNormalization(activation='tanh'))
	model.add(Dense(actions))
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	
	return model

def train_dnn():
	global REWARDS
	env = gym.make('ChessVsSelf-v0', render_mode='image', render_sampling=100)

	states = env.observation_space['board'].n + env.observation_space['player'].n
	actions = env.action_space.n

	model = build_dnc(states, actions)
	chess_agent = DNNChessAgent(model, env)
	agent = build_agent(model, chess_agent, states, actions, env)

	train_start_time = time.time()

	for i in range(EPISODES):
		start_time = time.time()

		state = env.reset()
		terminated = False

		sample = (i != 1 and (i - 1) % SAMPLING == 0) or i == 0

		while not terminated:
			action = agent.step(env, state)

			next_state = None
			reward = None
			terminated = None

			next_state, reward, terminated, truncated, info = env.step(action)

			if sample:
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

			env.render()

	agent.save_model()

	avg_rewards = np.array([np.mean(episode) for episode in REWARDS])

	plt.plot(agent.accuracy / np.linalg.norm(agent.accuracy))
	plt.plot(agent.loss / np.linalg.norm(agent.loss))
	plt.plot(avg_rewards)
	plt.title('Model Performance')
	plt.ylabel('Accuracy / Rewards')
	plt.xlabel('Epoch')
	plt.legend(['accuracy', 'loss', 'average rewards'], loc='upper left')
	plt.show()

	# with open('rewards.data', 'a+') as f:
	# 	f.write(str(REWARDS))

if __name__ == '__main__':

	#train_dnn()