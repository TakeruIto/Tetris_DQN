import numpy as np

def preprocess_state(game_state, board_height=22, board_width=10, num_piece_types=7):
    """
    Converts the raw game state from the Tetris environment into a 1D NumPy array.

    Args:
        game_state (tuple): A tuple containing (board_array, current_mino_obj, next_minos_list).
            - board_array (np.ndarray): 2D array (board_height x board_width) for landed blocks.
            - current_mino_obj (Mino): The current falling piece object.
            - next_minos_list (list): List of upcoming Mino objects.
        board_height (int): Height of the game board (visible part).
        board_width (int): Width of the game board (visible part).
        num_piece_types (int): Number of unique Tetris piece types.

    Returns:
        np.ndarray: A 1D NumPy array representing the concatenated features.
    """
    board_array, current_mino_obj, next_minos_list = game_state

    # 1. Normalize and Flatten Board
    # Ensure values are binary (0 for empty, 1 for occupied)
    normalized_board = np.where(board_array > 0, 1, 0)
    flattened_board = normalized_board.flatten()

    # 2. Represent Current Piece
    current_piece_board = np.zeros((board_height, board_width), dtype=int)
    if current_mino_obj:
        piece_shape = current_mino_obj.mino # The 2D array of the piece
        piece_h, piece_w = piece_shape.shape

        # Adjust coordinates:
        # current_mino_obj.x is for the board with walls (16 wide).
        # The visible board_array (22x10) starts at column 2 of the full board.
        # So, piece_col_on_visible_board = current_mino_obj.x - 2.
        # current_mino_obj.y is the row on the full board, which matches the visible board's row.
        piece_y_on_visible_board = current_mino_obj.y
        piece_x_on_visible_board = current_mino_obj.x - 2

        for r in range(piece_h):
            for c in range(piece_w):
                if piece_shape[r, c] > 0: # If it's part of the piece
                    board_r = piece_y_on_visible_board + r
                    board_c = piece_x_on_visible_board + c
                    # Check bounds before imprinting
                    if 0 <= board_r < board_height and 0 <= board_c < board_width:
                        current_piece_board[board_r, board_c] = 1
    
    flattened_current_piece = current_piece_board.flatten()

    # 3. Represent Next Piece (One-Hot Encoded Type)
    next_piece_one_hot = np.zeros(num_piece_types, dtype=int)
    if next_minos_list:
        next_mino = next_minos_list[0] # Get the first upcoming piece
        if next_mino and 0 <= next_mino.type < num_piece_types:
            next_piece_one_hot[next_mino.type] = 1
    
    # 4. Concatenate Features
    final_state_vector = np.concatenate([
        flattened_board,
        flattened_current_piece,
        next_piece_one_hot
    ])

    return final_state_vector
