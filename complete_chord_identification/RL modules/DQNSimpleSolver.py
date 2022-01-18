from collections import deque
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], enable=True)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import random
import tqdm
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from itertools import combinations
from datetime import datetime


def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = abs(error) > 1.0
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, "float32")
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


class DQNSolver:
    def __init__(
        self,
        env,
        n_episodes=1000,
        max_env_steps=None,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_log_decay=0.999,
        base_lr=0.001,
        max_lr=0.1,
        step_size=300,
        tau=0.125,
        alpha_decay=0.005,
        batch_size=64,
    ):
        self.env = env
        self.memory = deque(maxlen=500)
        self.epoch = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.tau = tau
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.best_loss = 8
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps

        # Init model
        state_input = Input(shape=self.env.observation_space.shape)
        h2 = Dense(48, activation="ReLU")(state_input)
        h3 = Dense(24, activation="ReLU")(h2)
        output = Dense(2, activation="linear")(h3)
        self.model = Model(inputs=state_input, outputs=output)
        adam = Adam()
        self.model.compile(loss=huber_loss, optimizer=adam)
        # Target model (Basically the same thing)
        state_input2 = Input(shape=self.env.observation_space.shape)
        h22 = Dense(48, activation="ReLU")(state_input2)
        h32 = Dense(24, activation="ReLU")(h22)
        output2 = Dense(2, activation="linear")(h32)
        self.target_model = Model(inputs=state_input2, outputs=output2)
        adam2 = Adam()
        self.target_model.compile(loss=huber_loss, optimizer=adam2)

    def masterScheduler(self, epoch):
        def scheduler(a, b):  # a,b, dummy here
            period = 2 * self.step_size
            cycle = math.floor(1 + epoch / period)
            x = abs(epoch / self.step_size - 2 * cycle + 1)
            delta = (self.max_lr - self.base_lr) * max(0, (1 - x))
            delta /= float(2 ** (cycle - 1))
            return self.base_lr + delta

        return scheduler

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        state = state.reshape((1, 12 * 2 + 1))
        return (
            self.env.action_space.sample()
            if (np.random.random() <= epsilon)
            else np.argmax(self.model.predict(state))
        )

    def get_epsilon(self, t):
        return max(
            self.epsilon_min,
            min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)),
        )

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), int(batch_size)))
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1, 12 * 2 + 1))
            next_state = next_state.reshape((1, 12 * 2 + 1))
            y_target = self.target_model.predict(state)
            y_target[0][action] = (
                reward
                if done
                else reward
                + self.gamma * np.max(self.target_model.predict(next_state)[0])
            )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        callback = tf.keras.callbacks.LearningRateScheduler(
            self.masterScheduler(self.epoch)
        )
        history = self.model.fit(
            np.array(x_batch),
            np.array(y_batch),
            batch_size=len(x_batch),
            verbose=0,
            callbacks=[callback],
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if random.random() < self.tau:
            self.target_model.set_weights(self.model.get_weights())
        return history.history["loss"]

    def run(self):
        # pbar = tqdm.tqdm(range(self.n_episodes))
        loss = []
        for e in range(self.n_episodes):
            self.epoch = e
            # pbar.set_description(f"previous loss = {loss[-1] if len(loss)>0 else 0}")
            done = False
            state = self.env.reset(random.randint(0, len(self.env.notes) - 1))
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            replayloss = self.replay(self.batch_size)
            loss.append(replayloss[0])
            if replayloss[0] < self.best_loss:
                self.model.save(f"dqnsimple_best_{replayloss[0]}")
                self.best_loss = replayloss[0]
            if e % 100 == 0 or e == self.n_episodes - 1:
                print(e)
                np.save("loss_dqnsimple.npy", np.array(loss))
        return loss
