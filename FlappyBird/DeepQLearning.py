from ple import PLE
from PIL import Image
import itertools
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from ple.games.flappybird import FlappyBird
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SDL_VIDEODRIVER"] = "dummy"

#Neural net architecture
def init_model():
	model = Sequential()
	model.add(Dense(512, activation="tanh"))
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

#Process the state dictionary to a list
def get_state(env):
	state_dict = env.getGameState()
	state = []
	for key in state_dict:
		state.append(state_dict[key])
	return np.asarray(state).reshape(1,len(state_dict))

#Convert the output order of the model to action in the environment
def convert_action(action_space, output):
	if output == 0:
		return action_space[0]
	else:
		return action_space[1]

def convert_position(action_space, actions):
	pos = np.zeros(actions.shape)
	for i in range(len(actions)):
		if actions[i] == action_space[1]:
			pos[i] = 1
	return pos

#Function to train the model from experience
def train_model(model, gamma, action_space, experiences):
	#Get information from experience memory
	exp_state = experiences[0][0]
	exp_action = experiences[0][1]
	exp_reward = experiences[0][2]
	exp_done = experiences[0][3]
	exp_next_state = experiences[0][4]
	for i in range(1, len(experiences)):
		exp_state = np.vstack((exp_state, experiences[i][0]))
		exp_action = np.vstack((exp_action, experiences[i][1]))
		exp_reward = np.vstack((exp_reward, experiences[i][2]))
		exp_done = np.vstack((exp_done, experiences[i][3]))
		exp_next_state = np.vstack((exp_next_state, experiences[i][4]))
	#Predict the Q values
	predicted_Qs = Q_function_approximation(model, exp_state)
	action_position = convert_position(action_space, exp_action)
	#TD target
	max_Qs_next_state = np.amax(Q_function_approximation(model, exp_next_state), axis=1)
	max_Qs_next_state = max_Qs_next_state.reshape((len(max_Qs_next_state), 1))
	updated_Q = exp_reward + gamma * max_Qs_next_state
	#If the next state is terminal, TD target is the reward
	for i in range(len(experiences)):
		if exp_reward[i] == 10:
			updated_Q[i] = 10
		elif exp_reward[i] == -10:
			updated_Q[i] = -10
	#Labels for traning
	labels = np.copy(predicted_Qs)
	for i in range(labels.shape[0]):
		labels[i][int(action_position[i][0])] = updated_Q[i]
	#Fit the model
	model.fit(exp_state, labels, batch_size = 32, verbose = 1)
	print("\n")

	#For debugging
	# after_training = Q_function_approximation(model, exp_state)
	# for i in range(len(experiences)):
	# 	if exp_reward[i] == -10:
	# 		print(predicted_Qs[i], end = " ")
	# 		print(action_position[i], end = " ")
	# 		print(exp_reward[i], end = " ")
	# 		print(updated_Q[i], end = " ")
	# 		print(labels[i], end = " ")
	# 		print(after_training[i])
	# for i in range(len(experiences)):
	# 	if exp_reward[i] == 10:
	# 		print(predicted_Qs[i], end = " ")
	# 		print(action_position[i], end = " ")
	# 		print(exp_reward[i], end = " ")
	# 		print(updated_Q[i], end = " ")
	# 		print(labels[i], end = " ")
	# 		print(after_training[i])
	# return model

def DeepQLearning(env, num_episodes, gamma=0.99, initial_epsilon=1, final_epsilon=0.001):
	#Find epsilon decay rate to get final_epsilon
	epsilon_decay = nth_root(num_episodes, final_epsilon/initial_epsilon)
	#Get action space
	action_space = env.getActionSet()
	#Initialize scores
	scores = []
	score = 0
	#Initilize experience memory
	experiences = []
	exp_size = 10000
	#Get the model and trained episodes
	if os.path.isfile("Fl4ppyB1rd.h5") and os.path.isfile("FlappyEpisodes.txt"):
		model = load_model("Fl4ppyB1rd.h5")
		file = open("FlappyEpisodes.txt", "r")
		start_episode = int(file.read())
		print("Found a model.")
	else:
		model = init_model()
		start_episode = 1
		print("No pre-trained model found")
	print("Start Training At Episode %d!" % start_episode)
	#Start the training
	for i in range(start_episode, num_episodes + 1):
		#Decay epsilon
		epsilon = initial_epsilon * (epsilon_decay ** i)
		#Save the model every 1000 episodes
		if i % 1000 == 0:
			#Save the model
			model.save("Fl4ppyB1rd.h5")
			file = open("FlappyEpisodes.txt", "w")
			file.write(str(i))
			file.close()
		#For every 100 episodes, calculate the avg score and train the model
		if i % 100 == 0:
			#Caculating
			if len(scores) != 0:
				total_scores = 0
				for scr in scores:
					total_scores += scr
				avg_scores = (total_scores * 1.0) / len(scores)
				scores.clear()
				#Print messages
				print("Episode %d/%d\nAvg reward last 100 episodes: %.3f\nCurrent Exp Memory Size: %d\nEpsilon: %.3f" % (i, num_episodes, avg_scores, len(experiences), epsilon))
			#Train the model
			if len(experiences) != 0:
				train_model(model,gamma, action_space, experiences)
				#Clear the experiences
				experiences.clear()
		#Reset Environment
		env.reset_game()
		state = get_state(env)
		score = 0
		#Generate an episode
		for t in itertools.count():
			#Make an action on the state image
			action = convert_action(action_space, epsilon_greedy(model, len(action_space), epsilon, state))
			#Get the reward in the next frame
			reward = env.act(action)
			score += reward
			#Get the next state
			next_state = get_state(env)
			#Save in experience memory
			experiences.append([state, action, reward, env.game_over(), next_state])
			#Save score when game over
			if env.game_over():
				scores.append(score)
				break
			#When exceed exp_size, train the model and free up memory
			if len(experiences) >= exp_size:
				print("Exceed Exp Memory at episode %d! Training and freeing up memory..." % i)
				train_model(model, gamma, action_space, experiences)
				experiences.clear()
			#Move to next state
			state = next_state
	return model

#Function to get the nth root of number n
def nth_root(num, n):
	return (n ** (1/num))

#Initialize the game environment
game = FlappyBird()
rewards = {"tick": 0, "positive" : 10, "loss" : -10}
env = PLE(game, fps=30, display_screen=False, reward_values=rewards)
env.init()

#Train Episodes
num_episodes = 100000

#Training
model = DeepQLearning(env, num_episodes)

#Save the model
model.save("Fl4ppyB1rd.h5")
print("Training completed!")