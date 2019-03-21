from ple import PLE
from PIL import Image
import itertools
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import load_model
from ple.games.flappybird import FlappyBird

def get_state(env):
	state_dict = env.getGameState()
	state = []
	for key in state_dict:
		state.append(state_dict[key])
	return np.asarray(state).reshape(1,len(state_dict))

def convert_action(action_space, output):
	if output == 0:
		return action_space[0]
	else:
		return action_space[1]

def show_samples(model, samples = 10):
	print("Solving Environment:")
	#Reinitialize the environment to enable display
	game = FlappyBird()
	rw = {"tick": 0, "positive" : 1, "loss" : 0}
	env = PLE(game, fps=30, display_screen=True, reward_values=rw)
	env.init()
	#Initilize some values
	total_reward = 0
	action_space = env.getActionSet()
	nothing = action_space[1]
	time.sleep(1)
	for i in range(samples):
		print("Sample %d" % (i+1), end = " ")
		time.sleep(0.5)
		env.reset_game()
		state = get_state(env)
		for t in itertools.count():
			time.sleep(0.01)
			#Make an action on the image
			action = convert_action(action_space, np.argmax(model.predict(state, verbose=0)))
			#Get the rewards in the frame
			reward = env.act(action)
			total_reward += reward
			#Get the next_state
			next_state = get_state(env)
			if env.game_over():
				print("Score: %d" % total_reward)
				total_reward = 0
				break
			state = next_state

def average_performance(model, num_episodes = 100):
	#Initialize the game environment
	game = FlappyBird()
	rw = {"tick": 0, "positive" : 1, "loss" : 0}
	env = PLE(game, fps=30, display_screen=False, reward_values=rw)
	env.init()
	print("Using trained model on %d episodes..." % num_episodes)
	action_space = env.getActionSet()
	nothing = action_space[1]
	rewards = []
	for i in range(num_episodes):
		env.reset_game()
		state = get_state(env)
		total_reward = 0
		for t in itertools.count():
			#Act on the image
			action = convert_action(action_space, np.argmax(model.predict(state, verbose=0)))
			#Get the rewards in the next frame
			reward = env.act(action)
			total_reward += reward
			#Get the next state
			next_state = get_state(env)
			if env.game_over():
				rewards.append(total_reward)
				total_reward = 0
				break
			state = next_state
	print("Average score in %d episodes: %.2f" % (num_episodes, (np.sum(rewards) * 1.0 / len(rewards))))

#Load the model
model = load_model("Fl4ppyB1rd.h5")

#Show average reward
average_performance(model)

#Show performances of the trained model
show_samples(model)

