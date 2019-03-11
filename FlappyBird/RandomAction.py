from ple import PLE
import random
import time
import itertools
from ple.games.flappybird import FlappyBird


game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)
action_space = env.getActionSet()

num_episodes = 10

env.init()
total_reward = 0

for i in range(1, num_episodes + 1):
	env.reset_game()
	print("Episode %d" % i)
	for i in itertools.count():
		state = env.getScreenRGB()
		if random.randint(0,1) == 0:
			action = action_space[0]
		else:
			action = action_space[1]
		reward = env.act(action)
		total_reward += reward
		if env.game_over():
			print("Total Reward: %d" % total_reward)
			total_reward = 0
			break
