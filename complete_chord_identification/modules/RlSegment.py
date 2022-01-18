import numpy as np
from music21 import *
import glob
import sys
import math
from tensorflow.keras.models import Sequential, Model, load_model

sys.path.append("../RL modules/")
from env_noOctave import SegmentationEnv


def rlsegment(piece):
    """
    rlsegment: returns a list of offset
    Input:
        string piece -- path string to the file location
    Returns:
        list of tuples (a,b): a is the start offset and b is the end offset of a segment
    """
    env = SegmentationEnv([piece])
    model = load_model("../results/dqn_normalized_2")
    offsets = []
    obs = env.reset(0)
    total_reward = 0
    max_reward = 0
    num_correct_segment = 0
    final_offset = env.offset[0][-1]
    while True:
        obs = obs.reshape((1, 12 * 2 + 1))
        action = np.argmax(model.predict(obs))
        if action == 1:
            offsets.append(env.current_noteoffset)
        obs, reward, done, info = env.step(action)
        # env.render()
        total_reward += reward
        max_reward += 1
        if reward == 1 and action == 1:
            num_correct_segment += 1
        if done:
            break
    toreturn = [(0, offsets[0])]
    for i in range(1, len(offsets)):
        toreturn.append((offsets[i - 1], offsets[i]))
    toreturn.append((offsets[-1], final_offset))
    return toreturn


if __name__ == "__main__":
    print(rlsegment("../musicxml(ok)/Prélude_in_A_Major.mxl"))
