import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Assuming dqn_agent.py is in the parent directory or accessible via PYTHONPATH
import sys
import os
# Add the parent directory to the sys.path to allow direct import of dqn_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dqn_agent import DQNAgent

# Disable TensorFlow interactive logging for cleaner test output
tf.keras.utils.disable_interactive_logging()

class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        """Set up a basic DQNAgent for each test."""
        self.state_shape = (535,) # (22*12 + 22*12 + 7)
        self.action_size = 7
        self.agent = DQNAgent(
            state_shape=self.state_shape,
            action_size=self.action_size,
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_rate=0.995,
            replay_buffer_size=1000, # Smaller buffer for tests
            batch_size=32,           # Smaller batch for tests
            target_update_freq=50    # Example frequency
        )

    def test_agent_creation_and_model_build(self):
        self.assertIsInstance(self.agent.q_network, Sequential, "Q-network should be a Keras Sequential model.")
        self.assertIsInstance(self.agent.target_network, Sequential, "Target network should be a Keras Sequential model.")

        # Check input shape of the Q-network
        # For a Dense layer with input_shape=(features,), the model's input_shape will be (None, features)
        self.assertEqual(self.agent.q_network.input_shape, (None, self.state_shape[0]),
                         "Q-network input shape is incorrect.")

        # Check output shape of the Q-network
        self.assertEqual(self.agent.q_network.output_shape, (None, self.action_size),
                         "Q-network output shape is incorrect.")

    def test_select_action(self):
        # Create a dummy state (1D array of the correct shape)
        dummy_state = np.random.rand(self.state_shape[0]).astype(np.float32)

        # Test action selection in training mode (epsilon > 0, so action can be random or greedy)
        action_train = self.agent.select_action(dummy_state, training=True)
        self.assertGreaterEqual(action_train, 0, "Action should be non-negative.")
        self.assertLess(action_train, self.action_size, "Action should be less than action_size.")

        # Test action selection in evaluation mode (epsilon = 0, should be greedy)
        self.agent.epsilon = 0.0 # Ensure greedy selection for this part
        action_eval_1 = self.agent.select_action(dummy_state, training=False) # or training=True with epsilon=0
        self.assertGreaterEqual(action_eval_1, 0)
        self.assertLess(action_eval_1, self.action_size)

        # For a given state and model weights, the greedy action should be deterministic
        # Re-running select_action (with training=False or epsilon=0) should yield the same action.
        action_eval_2 = self.agent.select_action(dummy_state, training=False)
        self.assertEqual(action_eval_1, action_eval_2, "Greedy action selection should be deterministic.")
        
        # Ensure training=False overrides any self.epsilon > 0 for greedy selection
        self.agent.epsilon = 1.0 # Set high epsilon
        action_eval_training_false = self.agent.select_action(dummy_state, training=False)
        action_eval_training_false_2 = self.agent.select_action(dummy_state, training=False)
        self.assertEqual(action_eval_training_false, action_eval_training_false_2, "Action with training=False should be deterministic.")


    def test_store_and_train(self):
        # Store a few dummy transitions
        num_transitions_to_add = self.agent.batch_size * 2 # Ensure enough for a batch
        for i in range(num_transitions_to_add):
            state = np.random.rand(self.state_shape[0]).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_shape[0]).astype(np.float32)
            done = (i % 5 == 0) # Some transitions are terminal
            self.agent.store_transition(state, action, reward, next_state, done)

        self.assertEqual(len(self.agent.replay_buffer), num_transitions_to_add,
                         "Buffer size incorrect after adding transitions.")

        # Test the train() method
        initial_epsilon = self.agent.epsilon
        try:
            self.agent.train() # This should run without errors
        except Exception as e:
            self.fail(f"agent.train() raised an exception: {e}")

        # Check if epsilon decayed (assuming train() calls decay_epsilon())
        if initial_epsilon > self.agent.epsilon_end:
             self.assertLess(self.agent.epsilon, initial_epsilon, "Epsilon should decay after training.")
        else:
             self.assertEqual(self.agent.epsilon, self.agent.epsilon_end, "Epsilon should be at its minimum.")

    def test_update_target_network(self):
        # Get initial weights
        q_network_weights_before = [np.copy(w) for w in self.agent.q_network.get_weights()]
        target_network_weights_before = [np.copy(w) for w in self.agent.target_network.get_weights()]

        # Ensure they are the same initially because setUp calls update_target_network via constructor
        for w_q, w_t in zip(q_network_weights_before, target_network_weights_before):
            np.testing.assert_array_equal(w_q, w_t, "Initial Q and Target weights should be identical.")

        # Simulate some training on the Q-network to change its weights
        # Add enough data for one batch and train
        if len(self.agent.replay_buffer) < self.agent.batch_size:
            for _ in range(self.agent.batch_size):
                state = np.random.rand(self.state_shape[0]).astype(np.float32)
                action = np.random.randint(0, self.action_size)
                reward = np.random.rand()
                next_state = np.random.rand(self.state_shape[0]).astype(np.float32)
                done = False
                self.agent.store_transition(state, action, reward, next_state, done)
        
        if len(self.agent.replay_buffer) >= self.agent.batch_size:
            self.agent.train() # This will modify self.agent.q_network weights

            q_network_weights_after_train = self.agent.q_network.get_weights()

            # Check that Q-network weights have changed from original target (unless training was a no-op)
            # This check can be flaky if training doesn't significantly change weights for a small batch.
            # A more robust way is to manually set weights if that's simpler than ensuring training changes them.
            # For simplicity, we'll assume train() does change them.
            # Test if at least one weight matrix is different
            changed = False
            for w_q_after, w_t_before in zip(q_network_weights_after_train, target_network_weights_before):
                if not np.array_equal(w_q_after, w_t_before):
                    changed = True
                    break
            self.assertTrue(changed, "Q-network weights should have changed after training step for this test.")


        # Call update_target_network
        self.agent.update_target_network()
        target_network_weights_after_update = self.agent.target_network.get_weights()

        # Verify that target network weights are now equal to the (potentially modified) Q-network weights
        q_network_weights_current = self.agent.q_network.get_weights() # Get current Q-weights again
        for w_q, w_t in zip(q_network_weights_current, target_network_weights_after_update):
            np.testing.assert_array_equal(w_q, w_t,
                                         "Target network weights should be equal to Q-network weights after update.")

if __name__ == '__main__':
    unittest.main()
