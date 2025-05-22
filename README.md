# Tetris_DQN

This repository hosts a Tetris game implemented in Python using OpenCV for display. While initially a simple Tetris clone, the codebase has been significantly refactored to support control by an AI agent, in addition to human play. The primary goal of this project is the implementation and demonstration of a Deep Q-Network (DQN) to play Tetris.

## Current Status

You can play Tetris yourself or watch a simple random AI agent play. The game is now structured to allow for the straightforward implementation of an AI agent (e.g., DQN). Key game logic has been separated from display and input, providing a clean environment for reinforcement learning experiments.

## Usage

To run the game, execute `main.py` from the command line.

```bash
python main.py [--mode <game_mode>]
```

**Command-Line Arguments:**

*   `--mode <game_mode>`: Specifies the game mode.
    *   `human`: Allows you to play the game manually. This is the default mode if the argument is omitted.
    *   `ai`: Runs a pre-programmed simple AI that chooses actions randomly.

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

**Controls (Human Mode):**

*   **Move Left:** Left Arrow or 'a'
*   **Move Right:** Right Arrow or 'd'
*   **Rotate Left:** 'z' key
*   **Rotate Right:** Up Arrow or 'x' key
*   **Soft Drop:** Down Arrow or 's' key
*   **Hard Drop:** Spacebar or 'w' key
*   **Quit Game:** ESC or 'e' key

## For AI Developers

The Tetris game has been designed to be a suitable environment for reinforcement learning agents.

### Key Components:

*   **`tetris.Tetris` class:** The main game environment.
    *   `reset() -> initial_state`: Resets the game to an initial state and returns it. Call this to begin a new episode.
    *   `step(action) -> (next_state, reward, done_flag)`: Executes the given `action` in the game.
        *   `action`: An integer representing one of the defined actions (see below).
        *   `next_state`: The new state of the game after the action, typically `(board, current_piece, next_pieces)`.
        *   `reward`: A numerical reward obtained from the last action.
        *   `done_flag`: A boolean that is `True` if the game is over, `False` otherwise.
    *   `get_state() -> (board, current_piece, next_pieces)`: Returns the current game state without taking a step.
        *   `board`: A NumPy array representing the game grid.
        *   `current_piece`: A `Mino` object for the currently falling piece.
        *   `next_pieces`: A list of `Mino` objects for upcoming pieces.

*   **Action Space:** Actions are defined as integer constants in `tetris.py`:
    *   `ACTION_MOVE_LEFT`
    *   `ACTION_MOVE_RIGHT`
    *   `ACTION_ROTATE_LEFT`
    *   `ACTION_ROTATE_RIGHT`
    *   `ACTION_DROP_SOFT`
    *   `ACTION_DROP_HARD`
    *   `ACTION_IDLE` (piece will attempt to fall one step due to gravity)

*   **`config.py`:** This file contains various configurable parameters for the game, including:
    *   Board dimensions (`BOARD_WIDTH`, `BOARD_HEIGHT`).
    *   Reward values for different events (e.g., `REWARD_LINE_CLEAR_SINGLE`, `REWARD_GAME_OVER`).
    *   Game settings like `AI_GAME_DELAY_MS`.
    Adjust these parameters to tune your AI's learning behavior.

*   **`display.py`:** Manages the game's visual output.
    *   `disable_rendering()`: Call this function before starting training loops to run the game in headless mode (no UI), which significantly speeds up training.
    *   `enable_rendering()`: Re-enables the display if needed.

## TODO

- [x] Stand up basic Tetris game.
- [x] Refactor code for AI agent integration (separate game logic, AI interface).
- [ ] Integrate a DQN agent with the refactored Tetris environment.
- [ ] Implement and train the DQN agent.
- [ ] Experiment with different reward structures and DQN architectures.
- [ ] Add comprehensive testing.
