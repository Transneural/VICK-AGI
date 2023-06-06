import gym
import tensorflow as tf
from keras import layers
import numpy as np
import random
from collections import deque
import os
import pickle
import json


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
       
    return config
config = load_config('config.json')



class ReinforcementLearningAgent:
    def __init__(
        self,
        env_name,
        pretrained_model=None,
        architecture=None,
        batch_size=64,
        memory_size=2000,
        use_ddqn=True,
        learning_rate=0.001,
        optimizer='adam',
        early_stopping=False,
        early_stopping_patience=3
    ):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.use_ddqn = use_ddqn
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.no_improvement = 0
        self.best_loss = float('inf')
        self.architecture = architecture
        self.model = self._build_model(pretrained_model)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self, pretrained_model=None):
        if pretrained_model is not None:
            # Load the pre-trained model and freeze its layers
            base_model = tf.keras.models.load_model(pretrained_model)
            base_model.trainable = False
            # Add additional layers for the reinforcement learning task
            model = tf.keras.Sequential()
            model.add(base_model)
            model.add(layers.Dense(24, activation='relu'))
            model.add(layers.Dense(self.action_size, activation='linear'))
        elif self.architecture is None:
            model = tf.keras.Sequential()
            model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(layers.Dense(24, activation='relu'))
            model.add(layers.Dense(self.action_size, activation='linear'))
        else:
            model = self.architecture(self.state_size, self.action_size)

        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate)

        model.compile(loss='mse', optimizer=optimizer)
        return model

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def preprocess_next_state(self, next_state):
        return self.preprocess_state(next_state)

    def get_reward(self, state, action, reward, next_state, done):
        return reward if not done else -10

    def get_done(self, state, action, reward, next_state, done):
        return done


    def update_target_model(self):
     self.target_model.set_weights(self.model.get_weights())
     
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                if self.use_ddqn:
                    next_action = np.argmax(self.model.predict(next_state)[0])
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][next_action]
                else:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000, render=False, weights_file=None, memory_file=None):
        config = load_config('config.json')

        # Check if the specified environment exists in the configuration
        for env_config in config['environments']:
            if env_config['name'] == self.env_name:
                model_path = env_config['model_path']
                # Load the pretrained model if available
                if os.path.exists(model_path):
                    self.model.load_weights(model_path)
                    self.update_target_model()
                    break
                else:
                    print(f"No pretrained model found for environment: {self.env_name}")
                    return
        else:
            print(f"Environment not found in the configuration: {self.env_name}")
            return

        if weights_file is not None and os.path.exists(weights_file):
            self.model.load_weights(weights_file)
            self.update_target_model()
        if memory_file is not None and os.path.exists(memory_file):
            with open(memory_file, 'rb') as f:
                self.memory = pickle.load(f)

        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                reward = reward if not done else -10
                self.remember(state, action, reward, next_state, done)
                state = next_state

            if len(self.memory) >= self.batch_size:
                self.replay()

            if e % 10 == 0:
                self.update_target_model()

            if weights_file is not None:
                self.model.save_weights(weights_file)

            if memory_file is not None:
                with open(memory_file, 'wb') as f:
                    pickle.dump(self.memory, f)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def get_score(self):
        scores = []
        for _ in range(10):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            score = 0
            while not done:
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_next_state(next_state)
                reward = self.get_reward(state, action, reward, next_state, done)
                done = self.get_done(state, action, reward, next_state, done)
                state = next_state
                score += reward
            scores.append(score)
        return np.mean(scores)


if __name__ == "__main__":
    agent = ReinforcementLearningAgent(env_name='CartPole-v1')
    agent.train()
