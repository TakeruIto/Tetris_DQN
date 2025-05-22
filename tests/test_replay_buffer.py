import unittest
import numpy as np
from collections import deque # For test_buffer_capacity verification

# Assuming dqn_agent.py is in the parent directory or accessible via PYTHONPATH
import sys
import os
# Add the parent directory to the sys.path to allow direct import of dqn_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dqn_agent import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def test_add_and_sample(self):
        buffer_capacity = 100
        replay_buffer = ReplayBuffer(max_size=buffer_capacity)

        # Add dummy transitions
        num_transitions = 10
        dummy_state_shape = (4,) # Example state shape
        for i in range(num_transitions):
            state = np.random.rand(*dummy_state_shape).astype(np.float32)
            action = np.random.randint(0, 5)
            reward = float(i)
            next_state = np.random.rand(*dummy_state_shape).astype(np.float32)
            done = (i % 2 == 0)
            replay_buffer.add(state, action, reward, next_state, done)

        self.assertEqual(len(replay_buffer), num_transitions, "Buffer size should match number of added transitions.")

        # Sample a batch
        batch_size = 5
        if num_transitions >= batch_size:
            batch = replay_buffer.sample(batch_size)
            self.assertEqual(len(batch), batch_size, "Sampled batch size should match requested batch size.")

            # Check types of elements in a sampled transition
            sampled_transition = batch[0]
            self.assertIsInstance(sampled_transition[0], np.ndarray, "State should be a NumPy array.")
            self.assertIsInstance(sampled_transition[1], int, "Action should be an integer.") # Assuming action is stored as int
            self.assertIsInstance(sampled_transition[2], float, "Reward should be a float.")
            self.assertIsInstance(sampled_transition[3], np.ndarray, "Next state should be a NumPy array.")
            self.assertIsInstance(sampled_transition[4], bool, "Done flag should be a boolean.")
        else:
            # Test sampling when buffer size is less than batch size (should raise error or return smaller batch)
            # The current ReplayBuffer.sample uses random.sample, which raises ValueError if batch_size > len(buffer)
            with self.assertRaises(ValueError, msg="Sampling more than buffer contains should raise ValueError."):
                replay_buffer.sample(batch_size)


    def test_buffer_capacity(self):
        buffer_capacity = 3
        replay_buffer = ReplayBuffer(max_size=buffer_capacity)

        # Add more items than capacity
        num_items_to_add = 5
        dummy_state_shape = (4,)
        added_rewards = [] # Keep track of rewards to verify FIFO

        for i in range(num_items_to_add):
            state = np.random.rand(*dummy_state_shape)
            action = i
            reward = float(i) # Use reward to track items
            next_state = np.random.rand(*dummy_state_shape)
            done = False
            replay_buffer.add(state, action, reward, next_state, done)
            added_rewards.append(reward)

        self.assertEqual(len(replay_buffer), buffer_capacity, "Buffer size should be equal to its max capacity.")

        # Verify that the oldest elements were discarded (FIFO)
        # The buffer should contain the last `buffer_capacity` items added.
        # In this case, rewards 2.0, 3.0, 4.0
        expected_rewards_in_buffer = added_rewards[-buffer_capacity:]
        
        # To check this, we need to look inside replay_buffer.buffer, which is a deque
        # This is a bit of white-box testing, but necessary to confirm FIFO.
        current_rewards_in_buffer = [transition[2] for transition in replay_buffer.buffer]
        
        self.assertListEqual(current_rewards_in_buffer, expected_rewards_in_buffer,
                             "Buffer should contain the latest items (FIFO behavior).")

if __name__ == '__main__':
    unittest.main()
