import math
from constants import SIM_WIDTH, SIM_HEIGHT, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT

def world_to_grid(world_x, world_y):
    """Converts world pixel coordinates to grid cell indices."""
    if world_x < 0 or world_x >= SIM_WIDTH or world_y < 0 or world_y >= SIM_HEIGHT:
        return None
    grid_x = int(world_x // GRID_CELL_SIZE)
    grid_y = int(world_y // GRID_CELL_SIZE)
    grid_x = max(0, min(GRID_WIDTH - 1, grid_x))
    grid_y = max(0, min(GRID_HEIGHT - 1, grid_y))
    return grid_x, grid_y

def grid_to_world_center(grid_x, grid_y):
    """Converts grid cell indices to the center world pixel coordinates of the cell."""
    world_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    world_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    return world_x, world_y

def clamp(value, min_val, max_val):
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))

# Can add more utility functions here later if needed