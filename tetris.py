import numpy as np
import cv2
# import display # display will be handled externally
from mino import Mino
import config # Import the configuration file

# CNT = 100 # Game speed will be handled by the agent or step frequency

# Action definitions
ACTION_MOVE_LEFT = 0
ACTION_MOVE_RIGHT = 1
ACTION_ROTATE_LEFT = 2
ACTION_ROTATE_RIGHT = 3
ACTION_DROP_SOFT = 4
ACTION_DROP_HARD = 5
ACTION_IDLE = 6


class Tetris():
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = self._init_board()
        self.minos = self._init_minos()
        self.score = self._init_score()
        self.rate = self._init_rate()
        self.chain = self._init_chain()
        # Additional state for AI
        self.current_piece = self.minos[0]
        self.next_pieces = self.minos[1:]
        self.game_over = False

    def _init_board(self):
        board = np.zeros((25, 16))
        board[:, 2] = 1  # Left wall
        board[:, -3] = 1 # Right wall
        board[21, :] = 1 # Bottom wall
        return board

    def _init_minos(self):
        # Generate a sequence of minos (current and next)
        return [Mino(5, 0, np.random.randint(7)) for _ in range(5)]

    def _init_score(self):
        return 0

    def _init_rate(self):
        return 1

    def _init_chain(self):
        return 0

    def get_state(self):
        """
        Returns the current game state for the AI.
        Includes the game board, current falling piece, and next piece(s).
        """
        # Normalize board representation if necessary, e.g., 0 for empty, 1 for block
        # For now, direct board values are fine.
        # Ensure piece information (type, rotation, position) is accessible.
        # current_piece_info = {
        # 'shape': self.current_piece.mino,
        # 'x': self.current_piece.x,
        # 'y': self.current_piece.y,
        # 'type': self.current_piece.type
        # }
        # next_pieces_info = [{
        # 'shape': p.mino, 'type': p.type
        # } for p in self.next_pieces]
        # return (self.board.copy(), current_piece_info, next_pieces_info)
        return (self.board[:22, 2:-2].copy(), self.current_piece, self.next_pieces)


    def step(self, action):
        """
        Applies an action, updates the game state, and returns state, reward, done flag.
        """
        if self.game_over:
            # Use configured game over reward
            return self.get_state(), config.REWARD_GAME_OVER, True

        reward = config.REWARD_SURVIVE_STEP  # Small reward for surviving the step
        done = False

        # 1. Apply player's action (movement, rotation, or initiating a drop)
        self._apply_action(action) # This updates self.current_piece state

        # 2. Handle piece falling (gravity or initiated drop)
        # If action is hard drop, piece is already at its lowest valid y by _apply_action.
        # If action is soft drop, piece moved down one step in _apply_action.
        # If action is idle or movement/rotation, piece needs to fall one step due to gravity.

        if action == ACTION_DROP_HARD:
            # Piece is already at its final y position (just above collision point)
            # Now, finalize it on the board.
            lines_cleared, score_gained_from_clear = self._update_board_and_score() # Now returns score too
            reward += score_gained_from_clear 
            reward += config.REWARD_HARD_DROP # Add reward for hard drop action
            if self._check_game_over():
                done = True
                reward = config.REWARD_GAME_OVER # Override reward if game over
            return self.get_state(), reward, done
        
        # For other actions (MOVE, ROTATE, IDLE, SOFT_DROP), attempt to move down one step due to gravity or soft drop
        self.current_piece.y += 1
        if self.current_piece.collision(self.board):
            self.current_piece.y -= 1 # Revert fall if collision
            # Piece has landed or is blocked by another piece after an action
            lines_cleared, score_gained_from_clear = self._update_board_and_score() # Finalize piece, check lines, spawn new
            reward += score_gained_from_clear
            if self._check_game_over():
                done = True
                reward = config.REWARD_GAME_OVER # Override reward if game over
        else:
            # Piece successfully moved down one step (either by soft drop or gravity)
            # Apply per-step cost if action wasn't a hard drop leading to finalization
            reward += config.REWARD_PER_STEP 
            # No collision yet, game continues.
            # If action was ACTION_DROP_SOFT, this counts as the soft drop.
            # No collision yet, game continues.
            # If action was ACTION_DROP_SOFT, this counts as the soft drop.
            # If action was IDLE, this is the gravity step.
            # If action was MOVE/ROTATE, this is gravity after the move/rotate.
            pass

        if self.game_over: # Should be set by _check_game_over if applicable
            done = True
        
        return self.get_state(), reward, done

    def _apply_action(self, action):
        """
        Helper method to apply the chosen action to the current piece.
        Updates piece position/rotation based on action.
        For ACTION_DROP_HARD, it moves the piece to the lowest possible valid position.
        For ACTION_DROP_SOFT, it moves the piece down one step.
        It does not finalize the piece on the board or handle gravity for IDLE/MOVE/ROTATE.
        """
        original_x = self.current_piece.x
        original_y = self.current_piece.y
        original_mino = self.current_piece.mino.copy()

        action_taken = True
        if action == ACTION_MOVE_LEFT:
            self.current_piece.x -= 1
        elif action == ACTION_MOVE_RIGHT:
            self.current_piece.x += 1
        elif action == ACTION_ROTATE_LEFT:
            self.current_piece.mino = np.rot90(self.current_piece.mino)
        elif action == ACTION_ROTATE_RIGHT:
            self.current_piece.mino = np.rot90(self.current_piece.mino, -1)
        elif action == ACTION_DROP_SOFT:
            # Soft drop is handled by the main step loop's gravity + collision check
            # Here, we just ensure it's a valid concept, but no immediate y change
            # The step() method will attempt self.current_piece.y += 1
            pass # The fall is handled in step()
        elif action == ACTION_DROP_HARD:
            # Move piece down until it's just above collision point
            # The actual placement on board happens in _update_board_and_score
            while not self.current_piece.collision(self.board):
                self.current_piece.y += 1
            self.current_piece.y -= 1 # Move back to last valid position
        elif action == ACTION_IDLE:
            # Idle action means let gravity handle the drop in the step() method
            action_taken = False # No specific piece manipulation here
            pass 
        
        # Revert if the action resulted in a collision (except for hard drop which handles its own y)
        if action != ACTION_DROP_HARD and self.current_piece.collision(self.board):
            self.current_piece.x = original_x
            self.current_piece.y = original_y
            self.current_piece.mino = original_mino


    def _update_board_and_score(self):
        """
        Called when a piece lands (collides after a downward movement).
        - Places the current piece onto the board.
        - Checks for and clears completed lines.
        - Updates the score and chain/rate.
        - Spawns the next piece.
        - Returns the number of lines cleared and the score gained from this clear.
        """
        h, w = self.current_piece.mino.shape
        self.board[self.current_piece.y:self.current_piece.y+h, self.current_piece.x:self.current_piece.x+w] += self.current_piece.mino
        
        lines_cleared = self._check_line_and_clear() # Renamed for clarity
        score_gained = 0
        
        if lines_cleared > 0:
            self.chain += 1
            if self.chain >= 3: # Combo bonus
                self.rate *= 1.1 # This rate affects actual game score, not AI reward directly here.
            
            # Use configured rewards for AI, actual score update for game
            if lines_cleared == 1:
                score_gained = config.REWARD_LINE_CLEAR_SINGLE
                self.score += self.rate * 10 # Original scoring for single
            elif lines_cleared == 2:
                score_gained = config.REWARD_LINE_CLEAR_DOUBLE
                self.score += self.rate * 30 # Original scoring for double
            elif lines_cleared == 3:
                score_gained = config.REWARD_LINE_CLEAR_TRIPLE
                self.score += self.rate * 60 # Original scoring for triple
            elif lines_cleared >= 4: # Tetris and above
                score_gained = config.REWARD_LINE_CLEAR_TETRIS
                self.score += self.rate * 100 # Original scoring for tetris

        else: # Line clear miss, reset chain and rate
            self.chain = self._init_chain()
            self.rate = self._init_rate()

        # Prepare next piece
        if len(self.minos) <= 1: # Should not happen if minos are replenished
             self.minos.extend([Mino(5,0,np.random.randint(7)) for _ in range(5)])

        self.minos.pop(0) # Remove the piece that just landed
        if not self.minos: # Replenish if somehow empty
             self.minos.extend([Mino(5,0,np.random.randint(7)) for _ in range(5)])
        self.minos.append(Mino(5, 0, np.random.randint(7))) # Add a new piece to the queue

        self.current_piece = self.minos[0]
        self.next_pieces = self.minos[1:5] # Ensure we always have a few next pieces visible
        
        return lines_cleared, score_gained


    def _check_line_and_clear(self): # Renamed from _check_line
        tmp = self.board[:21, 3:13] # Game area
        lines_to_clear_indices = np.where(np.all(tmp > 0, axis=1))[0]
        num_cleared = len(lines_to_clear_indices)

        if num_cleared > 0:
            # Remove filled lines by creating a new board segment without them
            remaining_lines = np.delete(tmp, lines_to_clear_indices, axis=0)
            # Add new empty lines at the top
            new_empty_lines = np.zeros((num_cleared, 10))
            self.board[:21, 3:13] = np.concatenate([new_empty_lines, remaining_lines])
            
        return num_cleared

    def _check_game_over(self):
        # Game over if the newly spawned piece immediately collides (top out)
        if self.current_piece.collision(self.board):
            self.game_over = True
            return True
        # Original condition: check if any blocks are in the buffer zone (row 1)
        # This condition might be redundant if the above handles top-out correctly.
        # However, it can catch scenarios where blocks are pushed into the spawn area.
        if np.any(self.board[1, 3:13] > 0):
             self.game_over = True
             return True
        self.game_over = False
        return False
