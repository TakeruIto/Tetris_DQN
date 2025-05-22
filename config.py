# config.py

# --- Board Dimensions ---
# Logical game width, excluding borders
BOARD_WIDTH = 10
# Logical game height, excluding the buffer zone at the top
# The main playable area is typically 20 rows high.
# tetris.py uses board[:21, 3:13] for line checks, which implies 21 rows.
# Let's use 20 as standard, can be adjusted if game logic implies 21 for active play.
BOARD_HEIGHT = 20


# --- Reward Values (for AI) ---
REWARD_LINE_CLEAR_SINGLE = 100
REWARD_LINE_CLEAR_DOUBLE = 300
REWARD_LINE_CLEAR_TRIPLE = 500
REWARD_LINE_CLEAR_TETRIS = 800  # Clearing 4 lines at once
REWARD_GAME_OVER = -500
REWARD_PER_STEP = -1  # Encourages efficiency; given for each piece dropped or game tick
REWARD_HARD_DROP = 10 # Small reward for committing to a piece placement via hard drop
REWARD_SURVIVE_STEP = 0.1 # Small reward for just surviving a step, used in tetris.py

# --- Game Speed / Control (for AI) ---
# Delay in milliseconds between AI steps when playing visibly (not headless training)
AI_GAME_DELAY_MS = 50


# --- Display Settings ---
# Pixel size of a single cell on the game board
CELL_SIZE_PX = 20  # Corresponds to W and H in display.py

# Width of the information panel (for score, next piece) in pixels
INFO_PANEL_WIDTH_PX = 100 # Current fixed value in display.py for the area next to the board

# --- Mino Settings ---
# Initial spawn position of a new mino.
# tetris.py uses Mino(5, 0, ...), where x=5 is roughly center for a 10-wide board (cols 3-12)
# Y=0 is the top-most row (buffer zone).
INITIAL_MINO_X = 5 # Logical X, to be translated to board array index
INITIAL_MINO_Y = 0 # Logical Y

# Number of next pieces to show in the display
NUM_NEXT_PIECES_DISPLAY = 4

# --- Game Mechanics ---
# Scoring system multipliers/constants if not directly tied to line clear rewards
# For example, if rate and chain bonuses were configurable:
# CHAIN_BONUS_RATE_INCREMENT = 0.1 # e.g. self.rate *= (1 + CHAIN_BONUS_RATE_INCREMENT)
# MIN_CHAIN_FOR_BONUS = 3

# --- AI Agent Specific ---
# (Could be moved to a separate AI config if it grows large)
# Example: Learning rate for a Q-learning agent
# AI_LEARNING_RATE = 0.001
# AI_DISCOUNT_FACTOR = 0.95

# --- File Paths ---
# Example: Path to save a trained AI model
# MODEL_SAVE_PATH = "models/tetris_ai_v1.pth"
# LOG_FILE_PATH = "logs/tetris_game.log"

# --- Debug ---
# DEBUG_MODE = False # General debug flag
# LOG_LEVEL = "INFO" # For a logging setup (e.g., "DEBUG", "INFO", "WARNING")
# PRINT_BOARD_ON_STEP = False # If true, print board to console for debugging without UI
# PRINT_REWARD_INFO = False # If true, print detailed reward info per step


# --- DQN Agent Hyperparameters ---
DQN_LEARNING_RATE = 0.001
DQN_DISCOUNT_FACTOR = 0.99  # gamma
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.01
DQN_EPSILON_DECAY_RATE = 0.9995 # Slower decay
DQN_REPLAY_BUFFER_SIZE = 20000
DQN_BATCH_SIZE = 64
DQN_TARGET_UPDATE_FREQ_EPISODES = 5 # In episodes
DQN_MODEL_SAVE_FREQ_EPISODES = 50   # In episodes
DQN_MODEL_PATH = "models/dqn_tetris.weights.h5"

# --- Training Settings ---
NUM_TRAINING_EPISODES = 2000
MAX_STEPS_PER_TRAINING_EPISODE = 3000


print("config.py loaded") # For debugging to ensure it's imported
