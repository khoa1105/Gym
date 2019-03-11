import gym
import sys
import time
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import itertools
from ple import PLE
from ple.games.flappybird import FlappyBird

def 