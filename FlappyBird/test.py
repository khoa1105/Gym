from ple import PLE
from PIL import Image
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from ple.games.flappybird import FlappyBird
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Convolutional Neural Net
def init_model():
	model = Sequential()
	model.add(Conv2D(32, 8, strides=4, padding='same', activation="relu", input_shape=(80,80,1)))
	model.add(Conv2D(64, 4, strides=2, padding='same', activation="relu"))
	model.add(Conv2D(64, 3, strides=1, padding='same', activation="relu"))
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dense(2, activation="linear"))
	model.compile(loss="mse", optimizer=Adam(lr=0.001))
	return model

#Use the conv net to estimate the Q values of each action
def Q_function_approximation(model, state):
	return model.predict(state, verbose=0)

#Epsilon greedy policy
def epsilon_greedy(model, nA, epsilon, state):
	A = np.ones(nA) * (epsilon/nA)
	bestA = np.argmax(Q_function_approximation(model, state))
	A[bestA] += (1-epsilon)
	action = np.random.choice(nA, p = A)
	return action

#Pre-process the input image
def get_resized_state(env):
	state = env.getScreenRGB()
	state = rgb2gray(state)
	state = resize(state, (1,80,80,1), anti_aliasing=True, mode='constant')
	return state

#Convert the output order of the model to action in the environment
def convert_action(action_space, output):
	if output == 0:
		return action_space[0]
	else:
		return action_space[1]

def convert_position(action_space, action):
	if action == action_space[0]:
		return 0
	else:
		return 1

def train_model(model, state, labels):
	model.fit(state, labels, verbose = 0)
	return model

def DeepQLearning(env, num_episodes, gamma=0.99, initial_epsilon=1):
	#Find epsilon decay rate so that epsilon after training is 0.005
	final_epsilon = 0.005
	epsilon_decay = nth_root(num_episodes, final_epsilon/initial_epsilon)
	#Get action space
	action_space = env.getActionSet()
	nothing = action_space[1]
	#Initialize scores
	scores = []
	score = 0
	#Get the model and trained episodes
	if os.path.isfile("Fl4ppyB0T.h5") and os.path.isfile("TrainedEpisodes.txt"):
		model = load_model("Fl4ppyB0T.h5")
		file = open("TrainedEpisodes.txt", "r")
		start_episode = int(file.read())
		print("Found a model.")
	else:
		model = init_model()
		start_episode = 1
		print("No pre-trained model found")
	print("Start Training At Episode %d!" % start_episode)
	#Start the training
	for i in range(start_episode, num_episodes + 1):
		#Save the model every 1000 episodes
		if i % 1000 == 0:
			model.save("Fl4ppyB0T.h5")
			file = open("TrainedEpisodes.txt", "w")
			file.write(str(i))
			file.close()
		#For every 100 episodes, calculate the avg score
		if i % 100 == 0:
			#Caculating
			if len(scores) != 0:
				total_scores = 0
				for scr in scores:
					total_scores += scr
				avg_scores = (total_scores * 1.0) / len(scores)
				scores.clear()
				#Print messages
				print("\rEpisode %d/%d. Avg reward last 100 episodes: %.2f" % (i, num_episodes, avg_scores), end = "")
				sys.stdout.flush()
		#Decay epsilon
		epsilon = initial_epsilon * (epsilon_decay ** i)
		#Reset Environment
		env.reset_game()
		state = get_resized_state(env)
		score = 0
		#Generate an episode
		for t in itertools.count():
			#Make an action on the state image
			action = convert_action(action_space, epsilon_greedy(model, len(action_space), epsilon, state))
			#Get the reward in the next 4 frames while doing nothing
			r1 = env.act(action)
			r2 = env.act(nothing)
			r3 = env.act(nothing)
			r4 = env.act(nothing)
			reward = r1 + r2 + r3 + r4
			score += reward
			#Get the next state
			next_state = get_resized_state(env)
			#Train the model
			predicted_Qs = Q_function_approximation(model, state)
			action_position = convert_position(action_space, action)
			updated_Q = reward + gamma * np.max(Q_function_approximation(model, next_state))
			labels = predicted_Qs
			labels[0][action_position] = updated_Q
			train_model(model, state, labels)
			if env.game_over():
				scores.append(score)
				break
			state = next_state
	return model

def nth_root(num, n):
	return (n ** (1/num))

#Initialize the game environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

#Train Episodes
num_episodes = 50000

#Training
model = DeepQLearning(env, num_episodes)

#Save the model
model.save("Fl4ppyB0T.h5")
print("Training completed!")
