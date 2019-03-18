from ple import PLE
from PIL import Image
import itertools
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import load_model
from ple.games.flappybird import FlappyBird

def get_resized_state(env):
	state = env.getScreenRGB()
	state = rgb2gray(state)
	state = resize(state, (1,80,80,1), anti_aliasing=True, mode='constant')
	return state

def convert_action(action_space, output):
	if output == 0:
		return action_space[0]
	else:
		return action_space[1]

def show_samples(env, model, samples = 10):
	print("Solving Environment:")
	#Reinitialize the environment to enable display
	game = FlappyBird()
	env = PLE(game, fps=30, display_screen=True)
	env.init()
	#Initilize some values
	total_reward = 0
	action_space = env.getActionSet()
	nothing = action_space[1]
	time.sleep(1)
	for i in range(samples):
		print("============")
		print("Sample %d" % (i+1))
		print("============")
		time.sleep(0.5)
		env.reset_game()
		state = get_resized_state(env)
		for t in itertools.count():
			time.sleep(0.01)
			#Make an action on the stacked image
			action = convert_action(action_space, np.argmax(model.predict(state, verbose=0)))
			print(action, end = " ")
			print(model.predict(state, verbose=0))
			r1 = env.act(action)
			r2 = env.act(nothing)
			r3 = env.act(nothing)
			r4 = env.act(nothing)
			#Get the next_state
			next_state = get_resized_state(env)
			#Sum the rewards
			reward = r1 + r2 + r3 + r4
			total_reward += reward
			if env.game_over():
				print("Total Reward: %d" % total_reward)
				total_reward = 0
				break
			state = next_state

def average_performance(env, model, num_episodes = 100):
	print("Using trained model on %d episodes..." % num_episodes)
	action_space = env.getActionSet()
	nothing = action_space[1]
	rewards = []
	for i in range(num_episodes):
		env.reset_game()
		state = get_resized_state(env)
		total_reward = 0
		for t in itertools.count():
			#Act on the image
			action = convert_action(action_space, np.argmax(model.predict(state, verbose=0)))
			r1 = env.act(action)
			r2 = env.act(nothing)
			r3 = env.act(nothing)
			r4 = env.act(nothing)
			#Get the next state
			next_state = get_resized_state(env)
			#Sum the rewards
			reward = r1 + r2 + r3 + r4
			total_reward += reward
			if env.game_over():
				rewards.append(total_reward)
				total_reward = 0
				break
			state = next_state
	print("Average Reward in %d episodes: %.2f" % (num_episodes, (np.sum(rewards) * 1.0 / len(rewards))))

#Initialize the game environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

#Load the model
model = load_model("Fl4ppyB0T.h5")

#Show average reward
#average_performance(env, model)

#Show performances of the trained model
show_samples(env, model)

