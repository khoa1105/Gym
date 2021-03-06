import gym
import sys
import time
from collections import defaultdict
import numpy as np
import itertools

def nth_root(num, n):
	return (n ** (1/num))

#Epsilon greedy policy
def epsilon_greedy(Q, nA, epsilon, state):
	A = np.ones(nA) * (epsilon/nA)
	bestA = np.argmax(Q[state])
	A[bestA] += (1-epsilon)
	action = np.random.choice(nA, p = A)
	return action

def SARSA(env, num_episodes, max_timesteps=200, alpha=0.85, gamma=0.99, epsilon=1):
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
		action = epsilon_greedy(Q, env.action_space.n, epsilon, state)
		for i in range(max_timesteps):
			next_state, reward, done, info = env.step(action)
			next_action = epsilon_greedy(Q, env.action_space.n, epsilon, next_state)
			#TD update
			Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
			if done:
				break
			action = next_action
			state = next_state
	print("\nTraining Completed!")
	time.sleep(1)
	return Q

#Find success rate of the algorithm
def show_success_rate(env, Q, episodes):
	print("Using trained model on %d episodes..." % episodes)
	time.sleep(1)
	succesful = 0
	for i in range(episodes):
		state = env.reset()
		for t in itertools.count():
			action = np.argmax(Q[state])
			next_state, reward, done, info = env.step(action)
			if done:
				if reward > 0:
					succesful += 1
				break
			state = next_state
	print("Success rate: %.2f%%" % ((succesful*100.0)/episodes))

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


#Create environment
env = gym.make('FrozenLake-v0')

#Learning parameters
num_episodes = 200000
test_episodes = 1000

#Solve the environment
Q = SARSA(env, num_episodes)

#Show success rate
show_success_rate(env, Q, test_episodes)

#Samples
#samples = 10
#show_samples(env, Q, samples)