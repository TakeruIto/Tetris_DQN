import numpy as np
import cv2

# --- Rendering Control ---
rendering_enabled = True
WINDOW_NAME = "Tetris" # Define a window name
WINDOW_HEIGHT = 400 # Placeholder, will be determined by board size
WINDOW_WIDTH = 300 # Placeholder, will be determined by board size + info panel
screen = None # Global screen variable to be initialized

def enable_rendering():
    """Enables display updates."""
    global rendering_enabled
    rendering_enabled = True
    # It might be good to also initialize display here if it wasn't
    # but for now, let's keep init_display separate.

def disable_rendering():
    """Disables display updates."""
    global rendering_enabled
    rendering_enabled = False

def is_rendering_enabled():
    """Checks if display updates are enabled."""
    return rendering_enabled

# --- Display Constants ---
W = 20  # Width of a single block cell in pixels
H = 20  # Height of a single block cell in pixels
COLORS = [[0, 0, 0],  # Background
          [128, 128, 128], # Wall
          [0, 191, 255],  # Mino Type 0 (e.g., I)
          [65, 105, 225],  # Mino Type 1 (e.g., J)
          [255, 165, 0],  # Mino Type 2 (e.g., L)
          [0, 255, 0],    # Mino Type 3 (e.g., O)
          [220, 20, 60],    # Mino Type 4 (e.g., S)
          [186, 85, 211], # Mino Type 5 (e.g., T)
          [255, 255, 0],  # Mino Type 6 (e.g., Z)
          ]

def init_display(title="Tetris"):
    """Initializes the display window if rendering is enabled."""
    global screen, WINDOW_NAME, WINDOW_HEIGHT, WINDOW_WIDTH
    WINDOW_NAME = title
    if rendering_enabled:
        # Calculate window size based on a typical board (e.g., 20_rows_visible x 10_cols_game_area)
        # Board dimensions are (22 rows visible, 10 cols game area)
        # tetris.py board[:22, 2:-2] -> board.shape (22,10)
        # So h=22, w=10 for the main game area.
        game_area_h = H * 22 
        game_area_w = W * 10
        info_panel_w = 100 # Width for score, next piece, etc.
        
        WINDOW_HEIGHT = game_area_h
        WINDOW_WIDTH = game_area_w + info_panel_w
        
        screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow(WINDOW_NAME)
        cv2.imshow(WINDOW_NAME, screen) # Show an initial black screen
        cv2.waitKey(1) # Necessary to render the window initially
    else:
        screen = None # Ensure screen is None if not rendering

def draw(board_state, minos_list, current_score):
    """Draws the game state if rendering is enabled."""
    global screen
    if not rendering_enabled or screen is None:
        # If called before init_display or rendering is off, screen might be None.
        # Or if init_display was called with rendering_enabled=false.
        if rendering_enabled and screen is None:
            # This case implies init_display was not called or failed.
            # For safety, one might call init_display here, or log a warning.
            # print("Warning: draw called before init_display or screen is None.")
            init_display() # Attempt to initialize if not done
            if screen is None: return # if still None, cannot draw
        elif not rendering_enabled:
            return

    # board_state is expected to be the visible part, e.g. board[:22, 2:-2]
    h, w = board_state.shape
    
    # Re-create canvas (screen) each time to clear previous frame
    # This is important because board_state changes.
    screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # Draw the game board cells
    for y_idx in range(h):
        for x_idx in range(w):
            color_idx = int(board_state[y_idx][x_idx])
            cv2.rectangle(screen, (x_idx*W, y_idx*H), ((x_idx+1)*W, (y_idx+1)*H),
                          COLORS[color_idx], thickness=-1)
            cv2.rectangle(screen, (x_idx*W, y_idx*H), ((x_idx+1)*W, (y_idx+1)*H),
                          (100, 100, 100), thickness=1) # Grid lines

    # Draw current falling mino and next minos
    if minos_list:
        _draw_mino(screen, minos_list[0], board_w_pixels=w*W) # Pass board width for positioning info panel
        if len(minos_list) > 1:
            _draw_next(screen, minos_list[1:], (w*W + 10, 10)) # Position next pieces in info panel

    _draw_score(screen, current_score, (w*W + 10, WINDOW_HEIGHT - 40)) # Position score at bottom of info panel
    
    # Draw the "death line" (usually above the visible part of the board, but shown for effect)
    # The actual game logic uses board[1,...] for game over, so line at H*2 is visual.
    cv2.line(screen, (0, H*2), (w*W, H*2), color=(255, 0, 0), thickness=2)
    
    # OpenCV uses BGR, so convert if COLORS are RGB (they are [0,0,0] style, typically BGR for cv2)
    # Assuming COLORS are already BGR or direct values for cv2.
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) # Not needed if COLORS are BGR
    
    cv2.imshow(WINDOW_NAME, screen)
    cv2.waitKey(1) # Essential for cv2.imshow to refresh


def _draw_mino(canvas, mino, board_w_pixels):
    # mino.x is relative to the full board (16 wide in tetris.py)
    # board_state passed to draw() is board[:, 2:-2], so it's 10 wide.
    # mino.x needs to be adjusted: mino.x - 2 (due to the 2-column left wall)
    display_x_offset = mino.x - 2 # Adjust for the walls not included in board_state
    display_y_offset = mino.y # y is usually from top
    
    mino_h, mino_w = mino.mino.shape
    for r_idx in range(mino_h):
        for c_idx in range(mino_w):
            if mino.mino[r_idx][c_idx]:
                color = COLORS[mino.type_ + 2] # +2 to offset background and wall colors
                cv2.rectangle(canvas,
                              ((display_x_offset + c_idx) * W, (display_y_offset + r_idx) * H),
                              ((display_x_offset + c_idx + 1) * W, (display_y_offset + r_idx + 1) * H),
                              color=color, thickness=-1)
                cv2.rectangle(canvas,
                              ((display_x_offset + c_idx) * W, (display_y_offset + r_idx) * H),
                              ((display_x_offset + c_idx + 1) * W, (display_y_offset + r_idx + 1) * H),
                              (100, 100, 100), thickness=1) # Grid lines for the mino itself


def _draw_next(canvas, next_minos_list, top_left_xy_info_panel):
    """Draws the upcoming pieces in the info panel."""
    sW, sH = 10, 10  # Smaller size for next pieces display
    panel_x, panel_y = top_left_xy_info_panel
    
    # Background for the "Next" area (optional)
    # cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + 80, panel_y + 320), (50,50,50), thickness=-1)
    cv2.putText(canvas, "NEXT:", (panel_x, panel_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for i, mino in enumerate(next_minos_list[:4]): # Display up to 4 next pieces
        if mino is None: continue
        mino_h, mino_w = mino.mino.shape
        
        # Center the small mino representation
        offset_x = int((80 - mino_w * sW) / 2) 
        base_y = panel_y + 30 + i * (sH * 4 + 20) # Spacing for each next mino

        for r_idx in range(mino_h):
            for c_idx in range(mino_w):
                if mino.mino[r_idx][c_idx]:
                    color = COLORS[mino.type_ + 2]
                    cv2.rectangle(canvas,
                                  (panel_x + offset_x + c_idx * sW, base_y + r_idx * sH),
                                  (panel_x + offset_x + (c_idx + 1) * sW, base_y + (r_idx + 1) * sH),
                                  color=color, thickness=-1)


def _draw_score(canvas, current_score, bottom_left_xy_info_panel):
    """Draws the score in the info panel."""
    panel_x, panel_y = bottom_left_xy_info_panel
    cv2.putText(canvas, "SCORE:", (panel_x, panel_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(canvas, str(current_score), (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), thickness=2)
