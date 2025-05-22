import os
import time
import numpy as np
import tensorflow as tf # For GPU check, optional

# Local imports
import tetris
from dqn_agent import DQNAgent # Assuming dqn_agent.py is in the same directory or PYTHONPATH
from utils.state_processor import preprocess_state # Assuming utils/state_processor.py
import display # From the original Tetris game, to disable rendering

# Attempt to import config, fall back to defaults if not found or specific params are missing
try:
    import config
except ImportError:
    config = None # No config file

# --- Constants and Hyperparameters ---
# Game and DQN parameters are now primarily sourced from config.py
# Fallback defaults are provided here if config or specific attributes are missing.

NUM_EPISODES = getattr(config, 'NUM_TRAINING_EPISODES', 1000)
MAX_STEPS_PER_EPISODE = getattr(config, 'MAX_STEPS_PER_TRAINING_EPISODE', 2000)
MODEL_PATH = getattr(config, 'DQN_MODEL_PATH', 'models/dqn_tetris.weights.h5') # Use new name
TARGET_UPDATE_FREQ_EPISODES = getattr(config, 'DQN_TARGET_UPDATE_FREQ_EPISODES', 10)
MODEL_SAVE_FREQ_EPISODES = getattr(config, 'DQN_MODEL_SAVE_FREQ_EPISODES', 50)

# DQN Hyperparameters from config, with fallbacks
LEARNING_RATE = getattr(config, 'DQN_LEARNING_RATE', 0.001)
DISCOUNT_FACTOR = getattr(config, 'DQN_DISCOUNT_FACTOR', 0.99)
EPSILON_START = getattr(config, 'DQN_EPSILON_START', 1.0)
EPSILON_END = getattr(config, 'DQN_EPSILON_END', 0.01)
EPSILON_DECAY_RATE = getattr(config, 'DQN_EPSILON_DECAY_RATE', 0.995)
REPLAY_BUFFER_SIZE = getattr(config, 'DQN_REPLAY_BUFFER_SIZE', 20000)
BATCH_SIZE = getattr(config, 'DQN_BATCH_SIZE', 64)


# Board and Action Dimensions (consistent with tetris.py and preprocess_state)
# These are more structural and less hyperparameter-like, can remain here or move to config too.
# For now, keep as is, assuming they are stable.
BOARD_HEIGHT = getattr(config, 'BOARD_HEIGHT', 22) # Visible board height (config.py uses 20 for play area)
# tetris.py get_state returns board[:22,...] so 22 is correct for the slice.
# utils.state_processor uses board_height=22 by default.
BOARD_WIDTH = getattr(config, 'BOARD_WIDTH', 10)  # Visible board width
NUM_PIECE_TYPES = 7 # Number of Tetris piece types (for one-hot encoding of next piece)
ACTION_SIZE = 7   # Number of possible actions (defined in tetris.py: 0-6)


# --- Main Training Script ---
def main():
    # 1. Initialization
    print("Initializing training...")
    
    # Ensure MODEL_PATH (potentially from config) is used for directory creation
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir): # Check if model_dir is not empty string
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory: {model_dir}")
    elif not model_dir: # Handle case where MODEL_PATH might be just a filename
        print("MODEL_PATH does not specify a directory. Model will be saved in the current directory.")


    # Disable Pygame rendering for faster training
    display.disable_rendering()
    print("Rendering disabled.")

    # Initialize Tetris environment
    env = tetris.Tetris()
    print("Tetris environment initialized.")

    # Calculate state size for the DQN agent
    # flattened_board_size = BOARD_HEIGHT * BOARD_WIDTH
    # flattened_current_piece_size = BOARD_HEIGHT * BOARD_WIDTH
    # next_piece_one_hot_size = NUM_PIECE_TYPES
    # state_size = flattened_board_size + flattened_current_piece_size + next_piece_one_hot_size
    # The preprocess_state function directly returns the state, so we can get its size from a sample
    _initial_raw_state_for_size_calc = env.get_state() # board is (22,10)
    _processed_state_for_size_calc = preprocess_state(
        _initial_raw_state_for_size_calc,
        board_height=BOARD_HEIGHT, # Should be 22 from env.get_state()
        board_width=BOARD_WIDTH,   # Should be 10 from env.get_state()
        num_piece_types=NUM_PIECE_TYPES
    )
    state_size = _processed_state_for_size_calc.shape[0]
    print(f"Calculated state size: {state_size} (using BOARD_HEIGHT={BOARD_HEIGHT}, BOARD_WIDTH={BOARD_WIDTH})")


    # Initialize DQNAgent
    agent = DQNAgent(
        state_shape=(state_size,), # Input shape for the neural network
        action_size=ACTION_SIZE,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_rate=EPSILON_DECAY_RATE, # Agent will handle its own decay logic
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=-1 # We will control target updates manually per episode
    )
    print("DQN Agent initialized.")

    # Optional: Load pre-trained weights
    if os.path.exists(MODEL_PATH) and os.path.isfile(MODEL_PATH): # Check if it's a file
        print(f"Loading pre-trained weights from {MODEL_PATH}...")
        try:
            agent.load_weights(MODEL_PATH)
        except Exception as e:
            print(f"Error loading weights: {e}. Starting from scratch.")
    else:
        print("No pre-trained weights found. Starting from scratch.")

    total_steps_across_episodes = 0
    training_start_time = time.time()

    # 2. Training Loop
    print(f"Starting training for {NUM_EPISODES} episodes...")
    for episode in range(1, NUM_EPISODES + 1):
        env.reset() # Reset the environment for a new episode
        raw_state = env.get_state()
        current_processed_state = preprocess_state(
            raw_state,
            board_height=BOARD_HEIGHT,
            board_width=BOARD_WIDTH,
            num_piece_types=NUM_PIECE_TYPES
        )
        episode_score = 0
        episode_steps = 0

        # Inner Loop (per episode)
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # Select action
            action = agent.select_action(current_processed_state, training=True)

            # Perform action in environment
            # env.step returns (next_state_tuple, reward, done_flag)
            # next_state_tuple is (board_array, current_mino_obj, next_minos_list)
            raw_next_state_tuple, reward, done = env.step(action)

            # Preprocess next state
            next_processed_state = preprocess_state(
                raw_next_state_tuple,
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                num_piece_types=NUM_PIECE_TYPES
            )
            # Store transition in replay buffer
            agent.store_transition(current_processed_state, action, reward, next_processed_state, done)

            # Update current state
            current_processed_state = next_processed_state
            episode_score += reward
            total_steps_across_episodes += 1
            episode_steps += 1

            # Call agent's training method (samples from buffer and trains)
            # The agent's train method also handles internal training_steps counter for its own target updates if configured
            agent.train()

            if done:
                break # End of episode

        # Epsilon decay is handled by the agent's train() method if called after each step or here if per episode
        # If agent.decay_epsilon() is designed to be called per episode:
        # agent.decay_epsilon() # This was in dqn_agent.py to be called within its train method.
        # If it's not, call it here. Assuming agent.train() calls decay_epsilon.

        # Update target network periodically (e.g., every N episodes)
        # Update target network periodically (using value from config)
        if episode % TARGET_UPDATE_FREQ_EPISODES == 0: # TARGET_UPDATE_FREQ_EPISODES from config
            agent.update_target_network()
            print(f"Episode {episode}: Target network updated.")

        # Log progress (using NUM_EPISODES from config)
        print(f"Episode: {episode}/{NUM_EPISODES} | Score: {episode_score:.2f} | Epsilon: {agent.epsilon:.4f} | Steps: {episode_steps} | Total Steps: {total_steps_across_episodes}")

        # Save model weights periodically (using MODEL_SAVE_FREQ_EPISODES and MODEL_PATH from config)
        if episode % MODEL_SAVE_FREQ_EPISODES == 0:
            agent.save_weights(MODEL_PATH) # MODEL_PATH from config
            print(f"Episode {episode}: Model weights saved to {MODEL_PATH}")

    # 3. After Loop
    print("\nTraining completed.")
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Total training time: {total_training_time:.2f} seconds.")

    # Save the final model (using MODEL_PATH from config)
    agent.save_weights(MODEL_PATH)
    print(f"Final model weights saved to {MODEL_PATH}")

    # Optional: Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {gpus}")
    else:
        print("No GPU detected by TensorFlow. Training on CPU.")

if __name__ == '__main__':
    main()
