import pygame
import random
from constants import (GRID_CELL_SIZE, OBSTACLE_COLOR, BASE_SIZE, BASE_COLOR,
                       RESOURCE_SIZE, RESOURCE_MAX_QUANTITY, RESOURCE_REGEN_RATE,
                       SIM_WIDTH, SIM_HEIGHT, DARK_GREY, YELLOW)
from utils import world_to_grid

class Obstacle:
    """Represents an impassable obstacle/wall on the grid."""
    def __init__(self, grid_x, grid_y):
        self.grid_pos = (grid_x, grid_y)
        self.rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        self.color = OBSTACLE_COLOR
        self.entity_type = "OBSTACLE" # Added for identification

    def draw(self, screen):
        if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Base:
    """Represents the central base."""
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.rect = pygame.Rect(x - BASE_SIZE // 2, y - BASE_SIZE // 2, BASE_SIZE, BASE_SIZE)
        self.grid_pos = world_to_grid(x, y)
        self.color = BASE_COLOR
        self.entity_type = "BASE"

    def draw(self, screen):
         if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Resource:
    """Represents a resource node."""
    _id_counter = 0
    def __init__(self, x, y):
        self.id = Resource._id_counter
        Resource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = world_to_grid(x, y)
        if self.grid_pos is None:
            raise ValueError(f"Resource created outside sim bounds: ({x},{y})")
        self.quantity = random.randint(RESOURCE_MAX_QUANTITY // 2, RESOURCE_MAX_QUANTITY)
        self.max_quantity = RESOURCE_MAX_QUANTITY
        self.regen_timer = 0
        self.rect = pygame.Rect(self.pos.x - RESOURCE_SIZE // 2, self.pos.y - RESOURCE_SIZE // 2, RESOURCE_SIZE, RESOURCE_SIZE)
        self.entity_type = "RESOURCE"

    def collect(self, amount=1):
        collected = min(amount, self.quantity)
        self.quantity -= collected
        self.quantity = max(0, self.quantity)
        return collected

    def update(self, dt):
        if self.quantity < self.max_quantity:
            self.regen_timer += dt
            regen_interval = 1.0 / RESOURCE_REGEN_RATE if RESOURCE_REGEN_RATE > 0 else float('inf')
            while self.regen_timer >= regen_interval and self.quantity < self.max_quantity: # Use while loop for faster regen catch-up
                self.quantity += 1
                self.regen_timer -= regen_interval
            self.quantity = min(self.max_quantity, self.quantity) # Ensure max isn't exceeded

    def draw(self, screen, is_targeted):
        if self.rect.right > SIM_WIDTH or self.rect.bottom > SIM_HEIGHT: return

        color = DARK_GREY
        if is_targeted: color = YELLOW
        elif self.quantity > 0:
            ratio = self.quantity / self.max_quantity
            red_val = int(255 * (1 - ratio))
            green_val = int(255 * ratio)
            color = (red_val, green_val, 0)
        pygame.draw.rect(screen, color, self.rect)