from ple import PLE
from PIL import Image
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
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
	model.add(Conv2D(32, 8, strides=4, padding='same', activation="relu"))
	model.add(Conv2D(64, 4, strides=2, padding='same', activation="relu"))
	model.add(Conv2D(64, 3, strides=1, padding='same', activation="relu"))
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dense(2, activation="linear"))
	model.compile(loss="mse", optimizer=Adam(lr=1e-6))
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

#Convert the output of model to action in the environment
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

#Remove experiences if exceed the size
def exp_check(experiences, size):
	while len(experiences) > size:
		experiences.remove(random.sample(experiences, 1))
	return experiences

def train_model(model, state, labels):
	model.fit(state, labels, verbose = 0)
	return model

def DeepQLearning(env, num_episodes, gamma=0.99, epsilon=1):
	#Find epsilon decay rate so that epsilon after training is 0.01
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)
	#Initialize function approximation model and experience
	model = init_model()
	#Get action space
	action_space = env.getActionSet()
	nothing = action_space[1]
	#Initialize scores
	scores = []
	score = 0
	#Experience Memory with size 3,000,000
	experiences = []
	exp_size = 3000000
	print("Start Training!")
	for i in range(1, num_episodes + 1):
		#Reset the score and print a message
		print("\rEpisode %d/%d. Reward last episode: %d." % (i, num_episodes, score), end = "")
		sys.stdout.flush()
		#For every 100 episodes, calculate the avg score
		if i % 100 == 0:
			#Caculating
			total_scores = 0
			for scr in scores:
				total_scores += scr
			avg_scores = (total_scores * 1.0) / len(scores)
			scores.clear()
			#Print messages
			print("\nAverage scores last 100 episodes: %.2f" % (avg_scores), end = "\n")
			sys.stdout.flush()
			#Start training if we have more than 300,000 experiences
			if len(experiences) > 300000:
				#Check if the amount of experiences exceed the pre-determined size
				experiences = exp_check(experiences, exp_size)
				#Random 300,000 experiences to train
				training_exp = random.sample(experiences, 300000)
				#Training
				for exp in training_exp:
					(state, action, reward, next_state) = (exp[0], exp[1], exp[2], exp[3])
					predicted_Qs = Q_function_approximation(model, state)
					action_position = convert_position(action_space, action)
					updated_Q = reward + gamma * np.max(Q_function_approximation(model, next_state))
					labels = predicted_Qs
					labels[0][action_position] = updated_Q
					train_model(model, state, labels)
		#Decay epsilon
		epsilon = epsilon * epsilon_decay
		#Reset Environment
		env.reset_game()
		score = 0
		for t in itertools.count():
			#Get the state
			state = get_resized_state(env)
			#Make an action on the image
			action = convert_action(action_space, epsilon_greedy(model, len(action_space), epsilon, state))
			reward = env.act(action)
			score += reward
			#Get the next state
			next_state = get_resized_state(env)
			#Store everything in Experience Memory
			experiences.append([state,action,reward,next_state])
			if env.game_over():
				scores.append(score)
				break
	return model

def nth_root(num, n):
	return (n ** (1/num))

#Initialize the game environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

#Episodes
num_episodes = 5000

#Training
model = DeepQLearning(env, num_episodes)

#Save the model
model.save("Fl4ppyB0T.h5")
print("Model saved!")