import pygame
import random
from constants import (GRID_CELL_SIZE, OBSTACLE_COLOR, BASE_SIZE, BASE_COLOR,
                       RESOURCE_SIZE, RESOURCE_MAX_QUANTITY, RESOURCE_REGEN_RATE,
                       WATER_SOURCE_SIZE, WATER_COLOR, THREAT_RADIUS, THREAT_COLOR,
                       SIM_WIDTH, SIM_HEIGHT, DARK_GREY, YELLOW)
from utils import world_to_grid

class Obstacle:
    """Represents an impassable obstacle/wall on the grid."""
    def __init__(self, grid_x, grid_y):
        self.grid_pos = (grid_x, grid_y)
        self.rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        self.color = OBSTACLE_COLOR
        self.entity_type = "OBSTACLE"

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
    """Represents a collectible resource node."""
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
            while self.regen_timer >= regen_interval and self.quantity < self.max_quantity:
                self.quantity += 1
                self.regen_timer -= regen_interval
            self.quantity = min(self.max_quantity, self.quantity)

    def draw(self, screen, is_targeted):
        if self.rect.right > SIM_WIDTH or self.rect.bottom > SIM_HEIGHT: return
        color = DARK_GREY
        if is_targeted: color = YELLOW
        elif self.quantity > 0:
            ratio = self.quantity / self.max_quantity
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0) # Red to Green
        pygame.draw.rect(screen, color, self.rect)

class WaterSource:
    """Represents a source of water."""
    _id_counter = 0
    def __init__(self, x, y):
        self.id = WaterSource._id_counter
        WaterSource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = world_to_grid(x, y)
        if self.grid_pos is None:
            raise ValueError(f"WaterSource created outside sim bounds: ({x},{y})")
        # Water sources are usually infinite or replenish very quickly
        self.quantity = float('inf') # Effectively infinite
        self.color = WATER_COLOR
        self.rect = pygame.Rect(self.pos.x - WATER_SOURCE_SIZE // 2, self.pos.y - WATER_SOURCE_SIZE // 2, WATER_SOURCE_SIZE, WATER_SOURCE_SIZE)
        self.entity_type = "WATER_SOURCE"

    def update(self, dt):
        pass # Does not deplete or regenerate currently

    def draw(self, screen):
        if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.ellipse(screen, self.color, self.rect) # Draw as ellipse/circle

class Threat:
    """Represents a simple threat source (e.g., predator location)."""
    _id_counter = 0
    def __init__(self, x, y):
        self.id = Threat._id_counter
        Threat._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.radius = THREAT_RADIUS # Radius within which agents feel fear
        # No grid_pos needed unless it moves on the grid
        # No rect needed unless it's selectable/collidable in quadtree
        self.color = THREAT_COLOR
        self.entity_type = "THREAT"
        # Basic rect for potential quadtree insertion / debug draw
        self.rect = pygame.Rect(x - 5, y - 5, 10, 10)

    def update(self, dt):
        pass # Static threat for now

    def draw(self, screen):
        """Draws a representation of the threat (e.g., for debugging)."""
        if 0 <= self.pos.x < SIM_WIDTH and 0 <= self.pos.y < SIM_HEIGHT:
            center = (int(self.pos.x), int(self.pos.y))
            pygame.draw.circle(screen, self.color, center, 8) # Draw a marker
            pygame.draw.circle(screen, self.color, center, int(self.radius), 1) # Draw fear radius