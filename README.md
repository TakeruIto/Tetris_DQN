# Tetris_DQN

This repository hosts a Tetris game implemented in Python using OpenCV for display. While initially a simple Tetris clone, the codebase has been significantly refactored and extended to support control by a Deep Q-Network (DQN) AI agent, in addition to human play and a basic random AI.

## Current Status

The project now includes a functional Deep Q-Network (DQN) agent capable of learning to play Tetris. You can train this agent from scratch, run a pre-trained model (if available), or play the game yourself. The game environment is structured to facilitate reinforcement learning experiments.

## Usage

To run the game, execute `main.py` from the command line.

```bash
python main.py [--mode <game_mode>]
```

**Command-Line Arguments:**

*   `--mode <game_mode>`: Specifies the game mode.
    *   `human`: Allows you to play the game manually. This is the default mode.
    *   `ai`: Runs a pre-programmed simple AI that chooses actions randomly.
    *   `dqn`: Runs the trained Deep Q-Network agent to play the game.

**Examples:**

*   Run in human play mode:
    ```bash
    python main.py --mode human
    ```
    or simply:
    ```bash
    python main.py
    ```
*   Run the AI example:
    ```bash
    python main.py --mode ai
    ```
*   Run the trained DQN agent:
    ```bash
    python main.py --mode dqn
    ```

## Controls (Human Mode)

*   **Move Left:** Left Arrow or 'a'
*   **Move Right:** Right Arrow or 'd'
*   **Rotate Left:** 'z' key
*   **Rotate Right:** Up Arrow or 'x' key
*   **Soft Drop:** Down Arrow or 's' key
*   **Hard Drop:** Spacebar or 'w' key
*   **Quit Game:** ESC or 'e' key

## AI Development and DQN Agent

The Tetris game has been designed to be a suitable environment for reinforcement learning agents.

### Key Environment Components:

*   **`tetris.Tetris` class:** The main game environment.
    *   `reset() -> initial_state`: Resets the game to an initial state and returns it. Call this to begin a new episode.
    *   `step(action) -> (next_state, reward, done_flag)`: Executes the given `action` in the game.
        *   `action`: An integer representing one of the defined actions (see below).
        *   `next_state`: The new state of the game after the action, typically `(board_array, current_mino_obj, next_minos_list)`.
        *   `reward`: A numerical reward obtained from the last action.
        *   `done_flag`: A boolean that is `True` if the game is over, `False` otherwise.
    *   `get_state() -> (board_array, current_mino_obj, next_minos_list)`: Returns the current game state without taking a step.
        *   `board_array`: A NumPy array representing the visible game grid (22 rows x 12 columns).
        *   `current_mino_obj`: A `Mino` object for the currently falling piece.
        *   `next_minos_list`: A list of `Mino` objects for upcoming pieces.

*   **Action Space:** Actions are defined as integer constants in `tetris.py`:
    *   `ACTION_MOVE_LEFT` (0)
    *   `ACTION_MOVE_RIGHT` (1)
    *   `ACTION_ROTATE_LEFT` (2)
    *   `ACTION_ROTATE_RIGHT` (3)
    *   `ACTION_DROP_SOFT` (4)
    *   `ACTION_DROP_HARD` (5)
    *   `ACTION_IDLE` (6) (piece will attempt to fall one step due to gravity)

*   **`config.py`:** This file contains various configurable parameters for the game and AI agent, including:
    *   Board dimensions (`BOARD_WIDTH`, `BOARD_HEIGHT`).
    *   Reward values for different events (e.g., `REWARD_LINE_CLEAR_SINGLE`, `REWARD_GAME_OVER`, `REWARD_PIECE_PLACED`).
    *   DQN hyperparameters and training settings.
    *   Game settings like `AI_GAME_DELAY_MS`.
    Adjust these parameters to tune your AI's learning behavior.

*   **`display.py`:** Manages the game's visual output using OpenCV.
    *   `disable_rendering()`: Call this function before starting training loops to run the game in headless mode (no UI), which significantly speeds up training.
    *   `enable_rendering()`: Re-enables the display if needed.

### DQN Agent Details:

*   **Implementation (`dqn_agent.py`)**:
    *   `DQNAgent` class: Implements the core DQN logic, including a Q-network (online network) and a target network for stable learning. It uses an epsilon-greedy policy for action selection during training, balancing exploration and exploitation.
    *   `ReplayBuffer` class: Stores experience tuples (state, action, reward, next_state, done) and allows for random sampling of mini-batches to train the Q-network, breaking correlations in the observed experience.

*   **State Representation (`utils/state_processor.py`)**:
    *   The `preprocess_state` function converts the raw game state into a feature vector suitable for the DQN.
    *   **Features include**:
        1.  A flattened representation of the visible game board (22 rows x 12 columns), normalized to 0s and 1s.
        2.  A flattened representation of the current falling piece projected onto an empty board of the same dimensions (22x12).
        3.  A one-hot encoded vector representing the type of the next upcoming piece (out of 7 types).
    *   **Resulting State Vector**: Concatenation of these features results in a 1D NumPy array of size 535 (22*12 + 22*12 + 7).

*   **Network Architecture**:
    *   The Q-network and target network are implemented as Keras Sequential models.
    *   The default architecture consists of Dense (fully connected) layers with ReLU activation functions and a final Dense layer with linear activation for outputting Q-values for each action.

*   **Hyperparameters and Rewards (`config.py`)**:
    *   Key DQN hyperparameters such as `DQN_LEARNING_RATE`, `DQN_DISCOUNT_FACTOR` (gamma), `DQN_EPSILON_START`, `DQN_EPSILON_END`, `DQN_EPSILON_DECAY_RATE`, `DQN_REPLAY_BUFFER_SIZE`, and `DQN_BATCH_SIZE` are defined in `config.py`.
    *   Reward values for game events (line clears, piece placement, game over) are also defined in `config.py`, allowing for experimentation with reward shaping.

### Training the DQN Agent:

*   **Script**: To train the DQN agent, run:
    ```bash
    python train_dqn.py
    ```

*   **Process**:
    *   The script initializes the `DQNAgent` and the Tetris environment.
    *   Training runs for a number of episodes specified by `NUM_TRAINING_EPISODES` in `config.py`.
    *   Game rendering is disabled by default during training to significantly improve speed.
    *   The agent's model weights (Q-network) are saved periodically to the path defined by `DQN_MODEL_PATH` in `config.py` (e.g., `models/dqn_tetris.weights.h5`).
    *   Progress, including episode scores and the current epsilon value, is logged to the console.
    *   Training a DQN agent can be computationally intensive and may take a significant amount of time depending on the number of episodes and hardware.

### Running the Trained DQN Agent:

*   **Script**: You can run a pre-trained (or currently training) agent using its dedicated script or through `main.py`:
    *   Standalone:
        ```bash
        python play_dqn.py
        ```
    *   Via `main.py`:
        ```bash
        python main.py --mode dqn
        ```

*   **Process**:
    *   The script loads the latest saved model weights from the path specified by `DQN_MODEL_PATH` in `config.py`.
    *   The game is run with rendering enabled, allowing you to watch the agent play.
    *   The agent uses a greedy policy for action selection (epsilon is set to 0), meaning it always chooses the action it believes to be optimal.
    *   The script reports the average score achieved by the agent over a number of evaluation episodes (configurable via `DQN_NUM_EVAL_EPISODES` in `config.py`).

## Running Tests

This project uses Python's built-in `unittest` framework for automated tests. Tests are located in the `tests/` directory.

To run all tests, navigate to the root directory of the repository and execute:
```bash
python -m unittest discover -s tests
```

## TODO

- [x] Stand up basic Tetris game.
- [x] Refactor code for AI agent integration (separate game logic, AI interface).
- [x] Integrate a DQN agent with the refactored Tetris environment (`dqn_agent.py`, `train_dqn.py`, `play_dqn.py`).
- [x] Implement the DQN agent and initial training setup.
- [~] Add comprehensive testing (core components like ReplayBuffer, StateProcessor, and DQNAgent structure are tested; more game interaction tests can be added).
- [ ] Experiment with different reward structures and DQN architectures.
- [ ] Hyperparameter tuning for optimal performance.
- [ ] Provide a pre-trained model.
- [ ] Add more detailed in-code documentation and comments.
