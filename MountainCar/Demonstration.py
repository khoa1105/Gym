import gym
import sys
import time
from collections import defaultdict
from keras.models import load_model
import numpy as np
import itertools

def show_samples(env, model, samples):
	print("Solving Environment:")
	total_reward = 0
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
			action = np.argmax(model.predict(state, verbose=0))
			next_state, reward, done, info = env.step(action)
			total_reward += reward
			if done:
				print("Total Reward: %d" % total_reward)
				total_reward = 0
				break
			state = next_state

def average_performance(env, model, num_episodes = 100):
	print("Using trained model on %d episodes..." % num_episodes)
	rewards = []
	for i in range(num_episodes):
		state = env.reset()
		total_reward = 0
		for t in itertools.count():
			state = np.asarray(state).reshape(1,2)
			action = np.argmax(model.predict(state, verbose=0))
			next_state, reward, done, info = env.step(action)
			total_reward += reward
			if done:
				rewards.append(total_reward)
				total_reward = 0
				break
			state = next_state

	print("Average Reward in %d episodes: %2f" % (num_episodes, (np.sum(rewards) * 1.0 / len(rewards))))

#Load the model
env = gym.make("MountainCar-v0")
model = load_model("MountainCar.h5")
num_samples = 10

#Show average reward
average_performance(env, model)

#Show performances of the trained model
show_samples(env, model, num_samples)


