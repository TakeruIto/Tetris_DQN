import argparse
import cv2
import random
import time # For AI play delay

from tetris import (
    Tetris,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
    ACTION_DROP_SOFT,
    ACTION_DROP_HARD,
    ACTION_IDLE
)
import display # For rendering the game
from play_dqn import run_dqn_evaluation # Add this import

# Key mappings for human play
KEY_MAP = {
    27: 'QUIT',        # ESC
    101: 'QUIT',       # 'e'
    2424832: ACTION_MOVE_LEFT,  # Left Arrow key code in cv2.waitKeyEx
    2555904: ACTION_MOVE_RIGHT, # Right Arrow key code in cv2.waitKeyEx
    2490368: ACTION_ROTATE_RIGHT, # Up Arrow key code in cv2.waitKeyEx (Let's use this for rotate right)
    # For consistency with previous, let's use 'a' and 'd' for move, 'z' 'x' for rotate
    ord('a'): ACTION_MOVE_LEFT,
    ord('d'): ACTION_MOVE_RIGHT,
    ord('z'): ACTION_ROTATE_LEFT, # 'z' for rotate left
    ord('x'): ACTION_ROTATE_RIGHT, # 'x' for rotate right
    ord('s'): ACTION_DROP_SOFT,    # 's' for soft drop
    2621440: ACTION_DROP_SOFT, # Down Arrow for soft drop
    32: ACTION_DROP_HARD,        # Spacebar for hard drop
    ord('w'): ACTION_DROP_HARD,    # 'w' for hard drop (alternative)
}

def human_play(tetris_game):
    """Runs the Tetris game for a human player."""
    display.init_display("Tetris - Human Play")
    tetris_game.reset()
    
    state, reward, done = tetris_game.get_state(), 0, False
    
    # Initial draw
    board_to_display = tetris_game.board[:22, 2:-2]
    display.draw(board_to_display, tetris_game.minos, tetris_game.score)

    while not done:
        action = ACTION_IDLE # Default action if no key is pressed or key is not mapped

        key = cv2.waitKeyEx(100) # Wait for 100ms for a key press

        if key != -1: # A key was pressed
            if key in KEY_MAP:
                command = KEY_MAP[key]
                if command == 'QUIT':
                    print("Quitting game.")
                    break
                action = command
            # else:
                # print(f"Unknown key: {key}") # For debugging new key codes

        # If no key was pressed, action remains ACTION_IDLE, leading to a soft drop/gravity
        # If a movement/rotation key was pressed, that action is taken.
        # Then, the step method handles the piece falling one step due to gravity.
        
        # For human play, we might want IDLE to truly do nothing and rely on explicit drops
        # However, the current step logic implies IDLE still involves gravity.
        # Let's make non-mapped keys or no key press result in a gravity step (IDLE)
        # and explicit drops are handled by their respective keys.

        next_state, reward, done = tetris_game.step(action)
        
        board_to_display = tetris_game.board[:22, 2:-2] # Get the updated board
        display.draw(board_to_display, tetris_game.minos, tetris_game.score)

        if done:
            print(f"Game Over! Final Score: {tetris_game.score}")
            # display.draw_game_over(tetris_game.score) # Assuming display has such a function
            cv2.putText(display.screen, "GAME OVER", (50, display.WINDOW_HEIGHT // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(display.screen, f"Score: {tetris_game.score}", (50, display.WINDOW_HEIGHT // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(display.WINDOW_NAME, display.screen)
            cv2.waitKey(0) # Wait indefinitely until a key is pressed to close
            break
        
        # Small delay to make the game playable, separate from cv2.waitKeyEx timeout
        # time.sleep(0.05) # Not strictly necessary if waitKeyEx has a timeout

def ai_play_example(tetris_game):
    """Runs a simple AI example (random actions)."""
    display.init_display("Tetris - AI Play Example")
    tetris_game.reset()

    state, reward, done = tetris_game.get_state(), 0, False
    
    board_to_display = tetris_game.board[:22, 2:-2]
    display.draw(board_to_display, tetris_game.minos, tetris_game.score)

    available_actions = [
        ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT,
        ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT,
        ACTION_DROP_SOFT, ACTION_DROP_HARD,
        ACTION_IDLE
    ]

    for _ in range(1000): # Limit AI play to 1000 steps for this example
        if done:
            break

        action = random.choice(available_actions)
        
        # print(f"AI Action: {action}") # For debugging AI actions

        next_state, reward, done = tetris_game.step(action)
        
        board_to_display = tetris_game.board[:22, 2:-2]
        display.draw(board_to_display, tetris_game.minos, tetris_game.score)
        
        cv2.waitKey(100) # 100ms delay to make AI visible

        if done:
            print(f"AI Game Over! Final Score: {tetris_game.score}")
            cv2.putText(display.screen, "AI GAME OVER", (30, display.WINDOW_HEIGHT // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(display.screen, f"Score: {tetris_game.score}", (30, display.WINDOW_HEIGHT // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(display.WINDOW_NAME, display.screen)
            cv2.waitKey(0) # Wait indefinitely
            break
    if not done:
        print("AI finished after 1000 steps.")


def main():
    parser = argparse.ArgumentParser(description="Play Tetris or watch an AI.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["human", "ai", "dqn"], # Add "dqn"
        default="human",
        help="Game mode: 'human' for manual play, 'ai' for AI example, 'dqn' for trained DQN agent."
    )
    args = parser.parse_args()

    # tetris_game = Tetris() # This might only be needed for human/ai_example

    if args.mode == "human":
        tetris_game = Tetris() # Initialize here
        human_play(tetris_game)
    elif args.mode == "ai":
        tetris_game = Tetris() # Initialize here
        ai_play_example(tetris_game)
    elif args.mode == "dqn":
        run_dqn_evaluation() # Call the imported function

    # cv2.destroyAllWindows() # This might be redundant if play_dqn.py handles its own windows.
    # If run_dqn_evaluation() calls destroyAllWindows(), then this one is not strictly needed when mode is dqn.
    # For human and ai modes, it's still relevant.
    # Let's keep it, as cv2.destroyAllWindows() is safe to call multiple times or if no windows exist.
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
