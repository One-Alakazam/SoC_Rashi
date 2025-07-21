import numpy as np
from helper import KungFu

class DQNAgent:
    def __init__(self, input_shape, num_actions, learning_rate):
        self.num_actions = num_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.learn_rate = learning_rate

        self.model = KungFu(input_shape, num_actions, learning_rate)
        self.target_model = KungFu(input_shape, num_actions, learning_rate)
        self.update_target_network()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def learn(self, memory):
        if len(memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        targets = q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        history = self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history["loss"][0]

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
