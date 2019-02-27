import gym

env = gym.make("MountainCar-v0")
print(env.state_space.n)