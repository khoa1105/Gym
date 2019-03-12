from ple import PLE
from PIL import Image
import itertools
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import load_model
from ple.games.flappybird import FlappyBird

def get_resized_state(env):
	state = env.getScreenRGB()
	state = rgb2gray(state)
	state = resize(state, (1,80,80,1), anti_aliasing=True, mode='constant')
	return state

def stack_image(img1, img2, img3):
	img = np.append(img1, img2, axis = 3)
	img = np.append(img,img3, axis = 3)
	return img

def show_samples(env, model, samples = 10):
	print("Solving Environment:")
	total_reward = 0
	time.sleep(1)
	for i in range(samples):
		print("============")
		print("Sample %d" % (i+1))
		print("============")
		time.sleep(0.5)
		env.reset_game()
		for t in itertools.count():
			time.sleep(0.01)
			#Get 3 consecutive states while doing nothing and stack them together
			state1 = get_resized_state(env)
			r1 = env.act(nothing)
			state2 = get_resized_state(env)
			r2 = env.act(nothing)
			state3 = get_resized_state(env)
			state = stack_image(state1, state2, state3)
			#Make an action on the stacked image
			action = np.argmax(model.predict(state, verbose=0))
			r3 = env.act(action)
			reward = r1 + r2 + r3
			#Sum the rewards
			total_reward += reward
			if env.game_over():
				print("Total Reward: %d" % total_reward)
				total_reward = 0
				break

def average_performance(env, model, num_episodes = 100):
	print("Using trained model on %d episodes..." % num_episodes)
	rewards = []
	for i in range(num_episodes):
		env.reset_game()
		total_reward = 0
		for t in itertools.count():
			#Get 3 consecutive states while doing nothing and stack them together
			state1 = get_resized_state(env)
			r1 = env.act(nothing)
			state2 = get_resized_state(env)
			r2 = env.act(nothing)
			state3 = get_resized_state(env)
			state = stack_image(state1, state2, state3)
			#Make an action on the stacked image
			action = np.argmax(model.predict(state, verbose=0))
			r3 = env.step(action)
			#Sum the rewards
			reward = r1 + r2 + r3
			total_reward += reward
			if done:
				rewards.append(total_reward)
				total_reward = 0
				break
	print("Average Reward in %d episodes: %2f" % (num_episodes, (np.sum(rewards) * 1.0 / len(rewards))))

#Initialize the game environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

#Load the model
model = load_model("Fl4ppyB0T.h5")

#Show average reward
average_performance(env, model)

#Show performances of the trained model
show_samples(env, model)

