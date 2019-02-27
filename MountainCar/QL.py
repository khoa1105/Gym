import gym
import sys
import time
from collections import defaultdict
import numpy as np
import itertools

#Function Approximation
def TD_update:

def Q_function_approximation(state):
	



#Epsilon greedy policy
def epsilon_greedy_with_FA(Q, nA, epsilon, state):
	A = np.ones(nA) * (epsilon/nA)
	bestA = np.argmax(Q_function_approximation(state))
	A[bestA] += (1-epsilon)
	action = np.random.choice(nA, p = A)
	return action

def DeepQLearning(env, num_episodes, max_timesteps=200, alpha=0.85, gamma=0.99, epsilon=1):
	#Find epsilon decay rate so that epsilon after training is 0.01
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)
	#Initialize Q
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	#Generate episodes
	print("Start Training!")
	time.sleep(0.5)
	for i in range(1, num_episodes+1):
		if i % 100 == 0:
			print("\rEpisode %d/%d" % (i, num_episodes), end = "")
			sys.stdout.flush()
		#Decay epsilon
		epsilon = epsilon * epsilon_decay
		#Reset enviroment
		state = env.reset()
		for i in range(max_timesteps):
			action = epsilon_greedy_with_FA(Q, env.action_space.n, epsilon, state)
			next_state, reward, done, info = env.step(action)
			best_next_action = np.argmax(Q_function_approximation(next_state))
			#TD update
			#Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
			if done:
				break
			state = next_state
	print("\nTraining Completed!")
	time.sleep(1)
	return Q

#Generate sample episodes
def show_samples(env, Q, samples):
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
			time.sleep(0.5)
			action = np.argmax(Q[state])
			next_state, reward, done, info = env.step(action)
			if done:
				break
			state = next_state

#Find nth root of a number
def nth_root(num, n):
	return (n ** (1/num))


#Create environment
env = gym.make('MountainCar-v0')

#Learning parameters
num_episodes = 2000

#Solve the environment
Q = DeepQLearning(env, num_episodes)

#Samples
samples = 10
show_samples(env, Q, samples)