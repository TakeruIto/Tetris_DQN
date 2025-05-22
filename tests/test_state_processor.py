import unittest
import numpy as np

# Assuming utils.state_processor and mino.py are accessible
# Add the parent directory to the sys.path to allow direct import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.state_processor import preprocess_state
from mino import Mino # Mino class is needed to create sample game states

class TestStateProcessor(unittest.TestCase):

    def setUp(self):
        # Standard dimensions for tests
        self.board_height = 22
        # The slice board[:22, 2:-2] from tetris.py results in a 12-column wide visible board
        # The preprocess_state function's board_width parameter refers to this visible width.
        self.board_width = 12 # Corrected from 10
        self.num_piece_types = 7

        # Expected shape: (board_flat + current_piece_flat + next_piece_one_hot)
        # (22*12 + 22*12 + 7) = (264 + 264 + 7) = 535
        self.expected_state_length = (self.board_height * self.board_width) * 2 + self.num_piece_types


    def test_preprocess_state_output_shape_and_type(self):
        # Create a sample raw game state
        # board_array is the visible part of the board, so 22x12
        board_array = np.zeros((self.board_height, self.board_width), dtype=int)
        
        # current_mino_obj: Mino(x, y, type)
        # x=5, y=0, type=0 (e.g., 'I' piece)
        # Note: current_mino_obj.x is relative to the full 16-wide board in tetris.py
        # preprocess_state expects it this way and adjusts with x-2 for the 10-wide visible board
        # If visible board is 12 wide (cols 2 to 13 of full board), then x=2 is first visible col.
        current_mino_obj = Mino(x=5, y=0, shape_type=0) 
        
        next_minos_list = [Mino(x=5, y=0, shape_type=1)]

        raw_state = (board_array, current_mino_obj, next_minos_list)

        processed_state = preprocess_state(
            raw_state,
            board_height=self.board_height,
            board_width=self.board_width, # Should be 12
            num_piece_types=self.num_piece_types
        )

        self.assertIsInstance(processed_state, np.ndarray, "Processed state should be a NumPy array.")
        self.assertEqual(processed_state.ndim, 1, "Processed state should be a 1D array.")
        self.assertEqual(processed_state.shape[0], self.expected_state_length,
                         f"Processed state length should be {self.expected_state_length}.")
        self.assertTrue(np.all((processed_state == 0) | (processed_state == 1)),
                        "All elements in processed_state should be 0 or 1.")


    def test_preprocess_state_board_representation(self):
        board_array = np.zeros((self.board_height, self.board_width), dtype=int)
        # Place a few known blocks
        board_array[self.board_height - 1, 0] = 1 # Bottom-left
        board_array[0, self.board_width - 1] = 2  # Top-right (will be normalized to 1)

        current_mino_obj = Mino(5, 0, 0) # Dummy piece, not focus of this test
        next_minos_list = [Mino(5, 0, 1)]

        raw_state = (board_array, current_mino_obj, next_minos_list)
        processed_state = preprocess_state(raw_state, self.board_height, self.board_width, self.num_piece_types)

        flattened_board_part = processed_state[:self.board_height * self.board_width]

        # Index for bottom-left: (self.board_height - 1) * self.board_width + 0
        self.assertEqual(flattened_board_part[(self.board_height - 1) * self.board_width + 0], 1,
                         "Bottom-left block not represented correctly.")
        # Index for top-right: 0 * self.board_width + (self.board_width - 1)
        self.assertEqual(flattened_board_part[self.board_width - 1], 1,
                         "Top-right block not represented correctly (should be normalized).")
        # Check a zero value
        self.assertEqual(flattened_board_part[15], 0, "An empty cell not represented correctly (index 15 for 12 wide).")


    def test_preprocess_state_current_piece_representation(self):
        board_array = np.zeros((self.board_height, self.board_width), dtype=int)
        
        # Test with 'O' piece (type 3 in mino.py)
        # 'O' piece: [[1,1],[1,1]] (actual values are piece_id)
        piece_type_O = 3
        # For a 12-wide visible board (indices 0-11), which maps to full board columns 2-13.
        # If Mino x=2, its leftmost part is at full_board_col=2, which is visible_col=0.
        mino_obj_O = Mino(x=2, y=0, shape_type=piece_type_O) 
        
        raw_state = (board_array, mino_obj_O, [Mino(5,0,0)]) # Next piece is dummy
        processed_state = preprocess_state(raw_state, self.board_height, self.board_width, self.num_piece_types)

        current_piece_offset = self.board_height * self.board_width
        flattened_current_piece_part = processed_state[current_piece_offset : current_piece_offset + (self.board_height * self.board_width)]

        # Check where the 'O' piece should be.
        # Mino x=2 (col 0 on visible), y=0 (row 0). Piece shape from mino.py for type 3 is [[4,4],[4,4]]
        # It should occupy (0,0), (0,1), (1,0), (1,1) on the current_piece_board
        expected_indices = [
            0 * self.board_width + 0,  # (0,0)
            0 * self.board_width + 1,  # (0,1)
            1 * self.board_width + 0,  # (1,0)
            1 * self.board_width + 1,  # (1,1)
        ]
        for idx in expected_indices:
            self.assertEqual(flattened_current_piece_part[idx], 1,
                             f"Current piece ('O') not represented correctly at index {idx}.")
        
        # Check an empty cell in the current piece representation
        empty_idx_check = 2 * self.board_width + 2 # e.g. (2,2)
        if empty_idx_check < len(flattened_current_piece_part):
             self.assertEqual(flattened_current_piece_part[empty_idx_check], 0,
                             "An empty cell in current piece board not represented correctly.")


    def test_preprocess_state_next_piece_representation(self):
        board_array = np.zeros((self.board_height, self.board_width), dtype=int)
        current_mino_obj = Mino(5, 0, 0) # Dummy
        
        next_piece_type_to_test = 2 # e.g., 'L' piece
        next_minos_list = [Mino(5, 0, next_piece_type_to_test)]

        raw_state = (board_array, current_mino_obj, next_minos_list)
        processed_state = preprocess_state(raw_state, self.board_height, self.board_width, self.num_piece_types)

        next_piece_offset = (self.board_height * self.board_width) * 2
        one_hot_next_piece_part = processed_state[next_piece_offset:]

        self.assertEqual(len(one_hot_next_piece_part), self.num_piece_types, "One-hot encoding length is wrong.")
        
        expected_one_hot = np.zeros(self.num_piece_types, dtype=int)
        expected_one_hot[next_piece_type_to_test] = 1
        
        np.testing.assert_array_equal(one_hot_next_piece_part, expected_one_hot,
                                      "Next piece one-hot encoding is incorrect.")

    def test_preprocess_state_no_next_piece(self):
        board_array = np.zeros((self.board_height, self.board_width), dtype=int)
        current_mino_obj = Mino(5, 0, 0)
        next_minos_list = [] # Empty list

        raw_state = (board_array, current_mino_obj, next_minos_list)
        processed_state = preprocess_state(raw_state, self.board_height, self.board_width, self.num_piece_types)

        next_piece_offset = (self.board_height * self.board_width) * 2
        one_hot_next_piece_part = processed_state[next_piece_offset:]
        
        expected_one_hot = np.zeros(self.num_piece_types, dtype=int) # Should be all zeros
        np.testing.assert_array_equal(one_hot_next_piece_part, expected_one_hot,
                                      "Next piece one-hot encoding should be all zeros if no next piece.")

if __name__ == '__main__':
    unittest.main()
