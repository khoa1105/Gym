import gym
import sys
import time
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import itertools

#Function Approximation: A 3-layers neural net with 2 input, 200 hidden, and 3 output units
def init_model(env):
	model = Sequential()
	model.add(Dense(100, input_dim=2, activation='relu'))
	model.add(Dense(env.action_space.n, activation='linear'))
	model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.001))
	return model

def TD_update(model, state, new_Q):
	model.fit(state, new_Q, verbose=0)
	return model

def Q_function_approximation(model, state):
	return model.predict(state, verbose=0)

#Epsilon greedy policy
def epsilon_greedy_with_FA(model, nA, epsilon, state):
	A = np.ones(nA) * (epsilon/nA)
	bestA = np.argmax(Q_function_approximation(model, state))
	A[bestA] += (1-epsilon)
	action = np.random.choice(nA, p = A)
	return action

def DeepQLearning(env, num_episodes, max_timesteps=200, gamma=0.99, epsilon=1):
	#Find epsilon decay rate so that epsilon after training is 0.01
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)
	#Initialize function approximation model and experience
	model = init_model(env)
	#Initialize scores
	scores = []
	#Generate episodes
	print("Start Training!")
	time.sleep(0.5)
	for i in range(1, num_episodes):
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
		#Decay epsilon
		epsilon = epsilon * epsilon_decay
		#Reset enviroment
		state = env.reset()
		highest_point = state[0]
		score = 0
		for i in range(max_timesteps):
			state = np.asarray(state).reshape(1,2)
			action = epsilon_greedy_with_FA(model, env.action_space.n, epsilon, state)
			next_state, reward, done, info = env.step(action)
			next_state = np.asarray(next_state).reshape(1,2)
			if next_state[0][0] > highest_point:
				reward += np.absolute(next_state[0][0] - highest_point) * 10
				highest_point = next_state[0][0]
			score += reward
			#Q(state, action) update
			predicted_Qs = Q_function_approximation(model, state)
			new_Q = reward + gamma * np.max(Q_function_approximation(model, next_state))
			labeled_Qs = predicted_Qs
			labeled_Qs[0][action] = new_Q
			TD_update(model, state, labeled_Qs)
			if done:
				scores.append(score)
				break
			state = next_state
	print("\nTraining Completed!")
	time.sleep(1)
	return model

#Generate sample episodes
def show_samples(env, model, samples):
	print("Solving Environment:")
	time.sleep(1)
	for i in range(samples):
		print("============")
		print("Sample %d" % (i+1))
		print("============")
		time.sleep(0.5)
		state = env.reset()
		for t in itertools.count():
			env.render()
			time.sleep(0.03)
			state = np.asarray(state).reshape(1,2)
			action = np.argmax(Q_function_approximation(model, state))
			print(action, end = " ")
			print(Q_function_approximation(model, state))
			next_state, reward, done, info = env.step(action)
			if done:
				break
			state = next_state

def show_success_rate(env, model, episodes):
	print("Using trained model on %d episodes..." % episodes)
	time.sleep(1)
	succesful = 0
	for i in range(episodes):
		state = env.reset()
		for t in itertools.count():
			state = np.asarray(state).reshape(1,2)
			action = np.argmax(Q_function_approximation(model, state))
			next_state, reward, done, info = env.step(action)
			if done:
				if reward == 1:
					succesful += 1
				break
			state = next_state
	print("Success rate: %.2f%%" % ((succesful*100.0)/episodes))


#Find nth root of a number
def nth_root(num, n):
	return (n ** (1/num))


#Create environment
env = gym.make('MountainCar-v0')

#Learning parameters
train_episodes = 50000
test_episodes = 100

#Solve the environment
model = DeepQLearning(env, train_episodes)

#Show success rate
#show_success_rate(env, model, test_episodes)

#Samples
samples = 10
show_samples(env, model, samples)