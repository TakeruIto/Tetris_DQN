import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    """
    A fixed-size replay buffer to store experience tuples.
    """
    def __init__(self, max_size):
        """
        Initialize the ReplayBuffer.

        Args:
            max_size (int): Maximum number of transitions to store in the buffer.
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience transition to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state observed.
            done (bool): Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for playing Tetris.
    """
    def __init__(self, state_shape, action_size, learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.995,
                 replay_buffer_size=10000, batch_size=64, target_update_freq=100):
        """
        Initialize the DQNAgent.

        Args:
            state_shape: Shape of the input state (e.g., (height, width, channels) or flattened).
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
            discount_factor (float): Gamma, discount factor for future rewards.
            epsilon_start (float): Initial value of epsilon for epsilon-greedy exploration.
            epsilon_end (float): Minimum value of epsilon.
            epsilon_decay_rate (float): Rate at which epsilon decays.
            replay_buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of the mini-batch for training.
            target_update_freq (int): Frequency (in training steps) for updating the target network.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Initialize Q-network and target network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()  # Initialize target network with Q-network weights

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.training_steps = 0 # Counter for target network updates

    def _build_model(self):
        """
        Build and compile the Q-network model.

        Returns:
            A Keras Sequential model.
        """
        model = Sequential()
        # Assuming state_shape is already flattened if it's a 2D board
        # If state_shape is like (rows, cols), a Flatten layer would be needed first
        # model.add(Flatten(input_shape=self.state_shape)) # Example if input is not flat
        model.add(Dense(128, activation='relu', input_shape=self.state_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output Q-values

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse')  # Mean Squared Error for Q-learning
        return model

    def update_target_network(self):
        """
        Copy weights from the Q-network to the target network.
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state, training=True):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state: The current preprocessed game state.
            training (bool): Whether the agent is in training mode. If False,
                             epsilon is effectively 0 (always greedy).

        Returns:
            The selected action (int).
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        else:
            # Ensure state is in the correct format for predict (e.g., batch of 1)
            if len(state.shape) == 1: # Assuming state is a flat array e.g. (100,)
                 state = np.expand_dims(state, axis=0) # Reshape to (1, 100)
            elif len(state.shape) == 2 and self.state_shape[0] == state.shape[0] and self.state_shape[1] == state.shape[1] and len(self.state_shape) == 2 : #e.g. (20,10)
                 state = np.expand_dims(state, axis=0) # Reshape to (1, 20,10)
            # Add more conditions if state can have other shapes e.g. (20,10,1)

            q_values = self.q_network.predict(state, verbose=0)
            return np.argmax(q_values[0])  # Exploit: action with max Q-value

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state observed.
            done (bool): Whether the episode has ended.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        """
        Train the Q-network using a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        minibatch = self.replay_buffer.sample(self.batch_size)

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Ensure states and next_states have the correct shape for the network
        # Example: if state_shape is (100,) and states is (64, 10, 10), it needs flattening
        # This might need adjustment based on how states are stored and preprocessed.
        # For now, assuming states and next_states are already in the correct flat shape (batch_size, feature_size)
        # If state_shape is (rows, cols), they might be (batch_size, rows, cols)
        # If the model has a Flatten layer as the first layer, it can handle (batch_size, rows, cols)
        # If the model expects (batch_size, features), and states are (batch_size, rows, cols),
        # then states = states.reshape(self.batch_size, -1) and similar for next_states.

        # Predict Q-values for current states using the Q-network
        current_q_values = self.q_network.predict(states, verbose=0)
        # Predict Q-values for next states using the target network
        next_q_values_target = self.target_network.predict(next_states, verbose=0)

        targets = np.copy(current_q_values)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values_target[i])

        # Train the Q-network
        self.q_network.fit(states, targets, epochs=1, verbose=0)

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

        self.decay_epsilon() # Decay epsilon after each training step

    def decay_epsilon(self):
        """
        Decay the epsilon value for exploration-exploitation balance.
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon_end, self.epsilon) # Ensure it doesn't go below min

    def load_weights(self, filepath):
        """
        Load model weights from a file.
        Args:
            filepath (str): Path to the HDF5 file containing the weights.
        """
        try:
            self.q_network.load_weights(filepath)
            self.update_target_network() # Also update target network
            print(f"Weights loaded successfully from {filepath}")
        except Exception as e:
            print(f"Error loading weights from {filepath}: {e}")


    def save_weights(self, filepath):
        """
        Save model weights to a file.
        Args:
            filepath (str): Path to save the HDF5 file for the weights.
        """
        try:
            self.q_network.save_weights(filepath)
            print(f"Weights saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving weights to {filepath}: {e}")
