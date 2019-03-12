from ple import PLE
from PIL import Image
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from ple.games.flappybird import FlappyBird

def init_model():
	model = Sequential()
	model.add(Conv2D(32, 8, strides=4, padding='same', activation="relu"))
	model.add(Conv2D(64, 4, strides=2, padding='same', activation="relu"))
	model.add(Conv2D(64, 3, strides=1, padding='same', activation="relu"))
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dense(2, activation="linear"))
	#model.compile(loss="mse", optimizer=Adam(lr=1e-6))
	return model

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
action_space = env.getActionSet()

env.init()
total_reward = 0

env.reset_game()

reward1 = env.act(action_space[0])
state1 = env.getScreenRGB()
image_resize1 = rgb2gray(state1)
image_resize1 = resize(image_resize1, (1,80,80,1), anti_aliasing = True, mode='constant')

reward2 = env.act(action_space[1])
state2 = env.getScreenRGB()
image_resize2 = rgb2gray(state2)
image_resize2 = resize(image_resize2, (1,80,80,1), anti_aliasing = True, mode='constant')

reward3 = env.act(action_space[1])
state3 = env.getScreenRGB()
image_resize3 = rgb2gray(state3)
image_resize3 = resize(image_resize3, (1,80,80,1), anti_aliasing = True, mode='constant')

image = np.append(image_resize1, image_resize2, axis = 3)
image = np.append(image, image_resize3, axis = 3)

print(image.shape)
#plt.imshow(image)
#plt.show()

#image = image_resize.reshape([1, 80, 80, 3])
#model = init_model()
#print(model.predict(image_resize, verbose = 0).shape)
