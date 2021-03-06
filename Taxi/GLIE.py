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

def monte_carlo_control(env, num_episodes, max_timesteps=100, gamma=1, epsilon=1):
	#Find epsilon decay rate so that epsilon after training is 0.01
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)
	#Initialize G, N, and Q
	return_sums = defaultdict(float)
	return_counts = defaultdict(float)
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	#Generate episodes
	print("Start Training!")
	time.sleep(0.5)
	for i in range(1, num_episodes+1):
		if i % 100 == 0:
			print("\rEpisode %d/%d" % (i, num_episodes), end = "")
			sys.stdout.flush()
		#decay_epsilon
		epsilon = epsilon * epsilon_decay
		#Record an episode
		episode = []
		state = env.reset()
		for i in range(max_timesteps):
			action = epsilon_greedy(Q, env.action_space.n, epsilon, state)
			next_state, reward, done, info = env.step(action)
			episode.append((state, action, reward))
			if done:
				break
			state = next_state

		#Learn from episode
		sa_in_episode = []
		for instance in episode:
			if (instance[0], instance[1]) not in sa_in_episode:
				sa_in_episode.append((instance[0], instance[1]))

		for state, action in sa_in_episode:
			sa_pair = (state, action)
			#First occurrence
			first_occurrence_index = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
			#Expected return
			G = sum([x[2] * (gamma ** i) for i,x in enumerate(episode[first_occurrence_index:])])
			#Update G, N, and Q
			return_sums[sa_pair] += G
			return_counts[sa_pair] += 1
			Q[state][action] = return_sums[sa_pair] / return_counts[sa_pair]
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
env = gym.make('Taxi-v2')

#Learning parameters
num_episodes = 200000
test_episodes = 200

#Solve the environment
Q = monte_carlo_control(env, num_episodes)

#Test the result
show_success_rate(env, Q, test_episodes)

#Generate sample episodes
#samples = 10
#show_samples(env, Q, samples)