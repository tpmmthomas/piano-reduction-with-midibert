from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from itertools import combinations
from datetime import datetime
from music21 import *
from stable_baselines3 import DQN
from collections import deque
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
import random
import tqdm
from env import SegmentationEnv
import glob
from DQNSolver import DQNSolver

testing_pieces = []

for piece in glob.glob("../data/testing/*"):
    testing_pieces.append(piece)

model = load_model("../results/DQN-normal")
print(model.summary())
env = SegmentationEnv(testing_pieces)
for i in range(len(testing_pieces)):
    obs = env.reset(i)
    # current_piece =
    total_reward = 0
    max_reward = 0
    num_correct_segment = 0
    while True:
        obs = obs.reshape((1, 12 * 7 + 1))
        action = np.argmax(model.predict(obs))
        obs, reward, done, info = env.step(action)
        # env.render()
        total_reward += reward
        max_reward += 1
        if reward == 1 and action == 1:
            num_correct_segment += 1
        if done:
            break
    print(
        f"Piece {testing_pieces[i]}, total reward = {total_reward}/{max_reward}, correct segment = {num_correct_segment}"
    )
