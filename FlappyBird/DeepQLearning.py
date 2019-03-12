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

def stack_image(img1, img2, img3):
	img = np.append(img1, img2, axis = 3)
	img = np.append(img,img3, axis = 3)
	return img

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
	#Experience Memory with size 3,000,000
	experiences = []
	exp_size = 3000000
	print("Start Training!")
	for i in range(1, num_episodes + 1):
		if i % 100 == 0:
			#Caculate average reward of the last 100 episodes
			total_scores = 0
			for scr in scores:
				total_scores += scr
			avg_scores = (total_scores * 1.0) / len(scores)
			scores.clear()
			#Print messages
			print("\rEpisode %d/%d. Average scores last 100 episodes: %.2f" % (i, num_episodes, avg_scores), end = "")
			sys.stdout.flush()
			#Start training if we have more than 300,000 experiences
			if len(experiences) > 300000:
				#Check if the amount of processed experiences exceed the pre-determined size
				experiences = exp_check(experiences, exp_size)
				#Random 300,000 experiences to train
				training_exp = random.sample(experiences, 300000)
				#Training
				for exp in training_exp:
					(state, action, reward, next_state) = (exp[0], exp[1], exp[2], exp[3])
					predicted_Qs = Q_function_approximation(model, state)
					updated_Q = reward + gamma * np.max(Q_function_approximation(model, next_state))
					labels = predicted_Qs
					labels[0][action] = updated_Q
					train_model(model, state, labels)
		#Decay epsilon
		epsilon = epsilon * epsilon_decay
		#Reset Environment
		env.reset_game()
		score = 0
		for t in itertools.count():
			#Get 3 consecutive states while doing nothing and stack them together
			state1 = get_resized_state(env)
			r1 = env.act(nothing)
			state2 = get_resized_state(env)
			r2 = env.act(nothing)
			state3 = get_resized_state(env)
			state = stack_image(state1, state2, state3)
			#Make an action on the stacked image
			action = epsilon_greedy(model, len(action_space), epsilon, state)
			r3 = env.act(action)
			reward = r1 + r2 + r3
			score += reward
			#Get the next state
			next_state = get_resized_state(env)
			#Stack the image of the future states
			next_state = stack_image(state2, state3, next_state)
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
num_episodes = 301

#Training
model = DeepQLearning(env, num_episodes)

#Save the model
model.save("Fl4ppyB0T.h5")