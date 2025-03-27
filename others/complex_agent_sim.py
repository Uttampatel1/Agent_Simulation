import pygame
import random
import math
import time
import heapq # For A* priority queue

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 750
GRID_CELL_SIZE = 20 # Size of each cell in the grid for pathfinding
GRID_WIDTH = SCREEN_WIDTH // GRID_CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_CELL_SIZE

AGENT_SIZE = 8
AGENT_SPEED = 60  # Pixels per second
RESOURCE_SIZE = 15
OBSTACLE_COLOR = (100, 100, 100)
BASE_SIZE = 40
BASE_COLOR = (200, 200, 0)

FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
GREY = (128, 128, 128)

# Agent States
STATE_IDLE = "IDLE"
STATE_MOVING_TO_RESOURCE = "MOVING_TO_RESOURCE"
STATE_MOVING_TO_BASE = "MOVING_TO_BASE"
STATE_COLLECTING = "COLLECTING"
STATE_RETURNING = "RETURNING" # Has resource, moving to base
STATE_EATING = "EATING"
STATE_RESTING = "RESTING"

# Needs Constants
MAX_ENERGY = 100
MAX_HUNGER = 100
ENERGY_DECAY_RATE = 2  # Per second
HUNGER_INCREASE_RATE = 3 # Per second
LOW_ENERGY_THRESHOLD = 25
HIGH_HUNGER_THRESHOLD = 75
EAT_AMOUNT = 50
REST_AMOUNT = 60
COLLECT_TIME = 1.5 # seconds
EAT_TIME = 2.0
REST_TIME = 3.0

# Resource Constants
RESOURCE_MAX_QUANTITY = 50
RESOURCE_REGEN_RATE = 0.5 # Quantity per second

# --- Helper Functions ---
def distance(pos1, pos2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def world_to_grid(world_x, world_y):
    """Converts world pixel coordinates to grid coordinates."""
    grid_x = int(world_x // GRID_CELL_SIZE)
    grid_y = int(world_y // GRID_CELL_SIZE)
    # Clamp to grid bounds
    grid_x = max(0, min(GRID_WIDTH - 1, grid_x))
    grid_y = max(0, min(GRID_HEIGHT - 1, grid_y))
    return grid_x, grid_y

def grid_to_world_center(grid_x, grid_y):
    """Converts grid coordinates to the center world pixel coordinates of the cell."""
    world_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    world_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    return world_x, world_y

def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- A* Pathfinding ---
def astar_pathfinding(grid, start_node, end_node):
    """
    Finds a path from start_node to end_node using A* on the grid.
    grid: 2D list where 0 is walkable, 1 is obstacle.
    start_node, end_node: tuples (grid_x, grid_y).
    Returns a list of grid nodes [(x1, y1), (x2, y2), ...] or None if no path.
    """
    if not grid or not (0 <= start_node[0] < len(grid[0])) or \
       not (0 <= start_node[1] < len(grid)) or \
       not (0 <= end_node[0] < len(grid[0])) or \
       not (0 <= end_node[1] < len(grid)) or \
       grid[start_node[1]][start_node[0]] == 1 or \
       grid[end_node[1]][end_node[0]] == 1:
        # print(f"A* Error: Invalid start/end node or node is obstacle. Start: {start_node}, End: {end_node}")
        return None # Invalid start or end, or start/end is obstacle

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4-directional movement
    # diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)] # Uncomment for 8-directional

    close_set = set()
    came_from = {}
    gscore = {start_node: 0}
    fscore = {start_node: heuristic(start_node, end_node)}
    oheap = [] # Priority queue (min heap)

    heapq.heappush(oheap, (fscore[start_node], start_node))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            # path.append(start_node) # Optional: include start node
            return path[::-1] # Return reversed path

        close_set.add(current)
        for i, j in neighbors: # Add diagonals here if using 8-way
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1 # Cost of moving to neighbor is 1

            # Check bounds
            if not (0 <= neighbor[0] < len(grid[0]) and 0 <= neighbor[1] < len(grid)):
                continue

            # Check if obstacle or already processed
            if grid[neighbor[1]][neighbor[0]] == 1 or neighbor in close_set:
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end_node)
                if neighbor not in [item[1] for item in oheap]: # Avoid duplicates in heap
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

    # print(f"A* Warning: No path found from {start_node} to {end_node}")
    return None # No path found


# --- Classes ---

class Obstacle:
    def __init__(self, grid_x, grid_y):
        self.grid_pos = (grid_x, grid_y)
        self.rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        self.color = OBSTACLE_COLOR

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Base:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.rect = pygame.Rect(x - BASE_SIZE // 2, y - BASE_SIZE // 2, BASE_SIZE, BASE_SIZE)
        self.grid_pos = world_to_grid(x,y)
        self.color = BASE_COLOR

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Resource:
    _id_counter = 0
    def __init__(self, x, y):
        self.id = Resource._id_counter
        Resource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = world_to_grid(x,y)
        self.quantity = random.randint(RESOURCE_MAX_QUANTITY // 2, RESOURCE_MAX_QUANTITY)
        self.max_quantity = RESOURCE_MAX_QUANTITY
        self.regen_timer = 0

    def collect(self, amount=1):
        collected = min(amount, self.quantity)
        self.quantity -= collected
        self.quantity = max(0, self.quantity) # Ensure non-negative
        return collected

    def update(self, dt):
        if self.quantity < self.max_quantity:
            self.regen_timer += dt
            if self.regen_timer >= 1.0 / RESOURCE_REGEN_RATE:
                self.quantity += 1
                self.quantity = min(self.max_quantity, self.quantity)
                self.regen_timer = 0 # Reset timer

    def draw(self, screen, is_targeted):
        # Color based on quantity and target status
        if is_targeted:
            color = YELLOW
        elif self.quantity == 0:
            color = GREY
        else:
            # Gradient from red (low) to green (high)
            ratio = self.quantity / self.max_quantity
            red_val = int(255 * (1 - ratio))
            green_val = int(255 * ratio)
            color = (red_val, green_val, 0)

        pygame.draw.rect(screen, color, (self.pos.x - RESOURCE_SIZE // 2, self.pos.y - RESOURCE_SIZE // 2, RESOURCE_SIZE, RESOURCE_SIZE))
        # Optional: Draw quantity text
        # font = pygame.font.SysFont(None, 18)
        # text = font.render(str(self.quantity), True, WHITE)
        # screen.blit(text, (self.pos.x + RESOURCE_SIZE/2, self.pos.y - RESOURCE_SIZE/2))

class Agent:
    _id_counter = 0
    def __init__(self, x, y, environment):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.environment = environment # Reference to the environment
        self.color = BLUE
        self.state = STATE_IDLE
        self.target_resource = None # Resource object currently targeted/being worked on
        self.target_pos_world = None # Current world coordinate destination (center of grid cell)
        self.current_path = [] # List of grid nodes (x, y) from A*
        self.path_index = 0

        # Needs
        self.energy = MAX_ENERGY
        self.hunger = 0
        self.carrying_resource = 0 # Amount of resource carried

        # Action Timers
        self.action_timer = 0

    def _find_path(self, target_grid_pos):
        """Calls A* and sets the path if found."""
        start_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        path = astar_pathfinding(self.environment.grid, start_grid_pos, target_grid_pos)
        if path:
            self.current_path = path
            self.path_index = 0
            if self.current_path:
                next_grid_node = self.current_path[self.path_index]
                self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            else: # Path has only one node (start=end) or is empty?
                 self.target_pos_world = None
                 self.current_path = []
            return True
        else:
            # print(f"Agent {self.id}: Pathfinding failed to {target_grid_pos}")
            self.current_path = []
            self.target_pos_world = None
            self.state = STATE_IDLE # Go idle if path fails
            # If we were targeting a resource, release it
            if self.target_resource:
                self.environment.mark_resource_available(self.target_resource)
                self.target_resource = None
            return False

    def _move_along_path(self, dt):
        """Moves the agent towards the current target_pos_world."""
        if not self.target_pos_world:
            # print(f"Agent {self.id}: Warning - Tried to move along path with no target_pos_world.")
            # If path exists but target is None, try setting next target
            if self.current_path and self.path_index < len(self.current_path):
                 next_grid_node = self.current_path[self.path_index]
                 self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            else: # No path or path finished
                 self.current_path = []
                 return False # Indicate movement didn't happen / path finished

        direction = self.target_pos_world - self.pos
        dist = direction.length()

        arrival_threshold = AGENT_SPEED * dt * 0.5 # Arrive if closer than half a step
        arrival_threshold = max(arrival_threshold, 2) # Minimum threshold

        if dist < arrival_threshold:
            # Reached the center of the current grid cell node
            self.pos = self.target_pos_world # Snap to target precisely
            self.path_index += 1
            if self.path_index < len(self.current_path):
                # Set next node in path as target
                next_grid_node = self.current_path[self.path_index]
                self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            else:
                # Reached end of path
                self.current_path = []
                self.target_pos_world = None
                return False # Indicate path is finished
        else:
            # Move towards target
            direction.normalize_ip()
            move_vector = direction * AGENT_SPEED * dt
            # Clamp move_vector if it overshoots
            if move_vector.length() > dist:
                self.pos += direction * dist # Move exactly to the target
            else:
                self.pos += move_vector

        # Basic boundary check (should ideally not be needed with pathfinding)
        self.pos.x = max(AGENT_SIZE//2, min(SCREEN_WIDTH - AGENT_SIZE//2, self.pos.x))
        self.pos.y = max(AGENT_SIZE//2, min(SCREEN_HEIGHT - AGENT_SIZE//2, self.pos.y))
        return True # Indicate still moving along path

    def update(self, dt):
        """Updates the agent's state, needs, and position."""
        # --- Needs Update ---
        self.energy -= ENERGY_DECAY_RATE * dt
        self.hunger += HUNGER_INCREASE_RATE * dt
        self.energy = max(0, self.energy)
        self.hunger = min(MAX_HUNGER, self.hunger)

        # --- High Priority Need Checks ---
        # Check needs *before* deciding on tasks
        if self.energy <= 0:
            print(f"Agent {self.id}: Died from exhaustion!")
            self.environment.remove_agent(self) # Agent dies
            return # Stop processing this agent

        needs_override = False
        # 1. Critical Hunger
        if self.hunger >= HIGH_HUNGER_THRESHOLD and self.state != STATE_EATING and self.state != STATE_MOVING_TO_BASE:
             if self.environment.base:
                 print(f"Agent {self.id}: Critically hungry! Moving to base to eat.")
                 if self._find_path(self.environment.base.grid_pos):
                     self.state = STATE_MOVING_TO_BASE
                     # Release resource if was going for one
                     if self.target_resource:
                         self.environment.mark_resource_available(self.target_resource)
                         self.target_resource = None
                 needs_override = True
             else: print(f"Agent {self.id}: Critically hungry but no base found!") # Should not happen if base exists

        # 2. Critical Energy (only override if not already resting/moving to rest)
        elif self.energy <= LOW_ENERGY_THRESHOLD and self.state != STATE_RESTING and self.state != STATE_MOVING_TO_BASE:
             if self.environment.base:
                 # Only override if not already handling hunger (higher priority)
                 if not needs_override:
                     print(f"Agent {self.id}: Critically tired! Moving to base to rest.")
                     if self._find_path(self.environment.base.grid_pos):
                         self.state = STATE_MOVING_TO_BASE
                         # Release resource if was going for one
                         if self.target_resource:
                             self.environment.mark_resource_available(self.target_resource)
                             self.target_resource = None
                     needs_override = True
             else: print(f"Agent {self.id}: Critically tired but no base found!")

        # --- State Machine (only if needs didn't override) ---
        if not needs_override:
            if self.state == STATE_IDLE:
                # What to do when idle?
                # 1. If carrying resources, return to base
                if self.carrying_resource > 0:
                    if self.environment.base:
                         print(f"Agent {self.id}: Carrying {self.carrying_resource}, returning to base.")
                         if self._find_path(self.environment.base.grid_pos):
                              self.state = STATE_RETURNING
                         else:
                             print(f"Agent {self.id}: Cannot find path to base to return resource!")
                             # Maybe drop resource? Or just wait? For now, wait.
                    else: print(f"Agent {self.id}: Carrying resource but no base found!")

                # 2. If not carrying, find a resource to collect
                else:
                    resource_target = self._find_nearest_available_resource()
                    if resource_target:
                        print(f"Agent {self.id}: Found resource {resource_target.id}, moving to collect.")
                        if self._find_path(resource_target.grid_pos):
                            self.state = STATE_MOVING_TO_RESOURCE
                            self.target_resource = resource_target
                            self.environment.mark_resource_targeted(resource_target, self)
                        else:
                            # Pathfinding failed, remain idle
                            self.target_resource = None # Ensure no target if path failed
                    # else: No available resources, remain idle


            elif self.state == STATE_MOVING_TO_RESOURCE:
                path_finished = not self._move_along_path(dt)
                if path_finished:
                    # Arrived at resource location
                    if self.target_resource and world_to_grid(self.pos.x, self.pos.y) == self.target_resource.grid_pos:
                        print(f"Agent {self.id}: Arrived at resource {self.target_resource.id}.")
                        # Check if resource still has quantity AND is still targeted by *this* agent
                        if self.target_resource.quantity > 0 and self.environment.is_resource_targeted_by(self.target_resource, self):
                             self.state = STATE_COLLECTING
                             self.action_timer = COLLECT_TIME
                        else:
                             print(f"Agent {self.id}: Resource {self.target_resource.id} empty or taken upon arrival.")
                             self.environment.mark_resource_available(self.target_resource) # Make sure it's marked available if we didn't collect
                             self.target_resource = None
                             self.state = STATE_IDLE
                    else:
                        # Path finished, but maybe resource disappeared or target changed? Go idle.
                        print(f"Agent {self.id}: Finished path but target resource invalid/missing.")
                        if self.target_resource: # Release if we still thought we had one
                            self.environment.mark_resource_available(self.target_resource)
                        self.target_resource = None
                        self.state = STATE_IDLE


            elif self.state == STATE_COLLECTING:
                self.action_timer -= dt
                if self.action_timer <= 0:
                    if self.target_resource and self.target_resource.quantity > 0:
                        collected = self.target_resource.collect(1)
                        self.carrying_resource += collected
                        print(f"Agent {self.id}: Collected 1 unit from {self.target_resource.id}. Now carrying {self.carrying_resource}. Resource left: {self.target_resource.quantity}")

                        # Decide what next: Collect more or return?
                        # Simple logic: return if resource empty or agent has enough energy/hunger margin
                        # More complex: Carry capacity, distance to base etc.
                        if self.target_resource.quantity == 0 or self.energy < LOW_ENERGY_THRESHOLD + 10 or self.hunger > HIGH_HUNGER_THRESHOLD - 10:
                            print(f"Agent {self.id}: Finished collecting (Resource empty or needs pressing). Returning to base.")
                            self.environment.mark_resource_available(self.target_resource)
                            self.target_resource = None
                            if self.environment.base:
                                if self._find_path(self.environment.base.grid_pos):
                                    self.state = STATE_RETURNING
                                else:
                                    print(f"Agent {self.id}: Cannot find path to base!")
                                    self.state = STATE_IDLE # Go idle if path fails
                            else:
                                print(f"Agent {self.id}: No base to return to!")
                                self.state = STATE_IDLE
                        else:
                            # Continue collecting
                             self.action_timer = COLLECT_TIME # Reset timer for next unit
                    else:
                        # Resource depleted while waiting or target lost
                        print(f"Agent {self.id}: Resource {self.target_resource.id if self.target_resource else '?'} depleted or lost during collection.")
                        if self.target_resource:
                            self.environment.mark_resource_available(self.target_resource)
                            self.target_resource = None
                        # If carrying something, try returning, else go idle
                        if self.carrying_resource > 0 and self.environment.base:
                             if self._find_path(self.environment.base.grid_pos):
                                 self.state = STATE_RETURNING
                             else: self.state = STATE_IDLE
                        else:
                             self.state = STATE_IDLE


            elif self.state == STATE_RETURNING: # Moving to base while carrying resources
                path_finished = not self._move_along_path(dt)
                if path_finished:
                     # Arrived at base location
                     if self.environment.base and world_to_grid(self.pos.x, self.pos.y) == self.environment.base.grid_pos:
                        print(f"Agent {self.id}: Arrived at base, dropping {self.carrying_resource} resources.")
                        self.environment.add_base_resources(self.carrying_resource) # Add to base total
                        self.carrying_resource = 0
                        # Decide: Eat? Rest? Go get more? Based on needs.
                        if self.hunger >= HIGH_HUNGER_THRESHOLD:
                            self.state = STATE_EATING
                            self.action_timer = EAT_TIME
                        elif self.energy <= LOW_ENERGY_THRESHOLD:
                             self.state = STATE_RESTING
                             self.action_timer = REST_TIME
                        else:
                            self.state = STATE_IDLE # Go look for more resources
                     else:
                         # Path finished but not at base? Should not happen unless base moved/path error
                         print(f"Agent {self.id}: Finished return path but not at base.")
                         self.state = STATE_IDLE # Go idle and try again


            elif self.state == STATE_MOVING_TO_BASE: # Moving to base for needs (hunger/energy)
                 path_finished = not self._move_along_path(dt)
                 if path_finished:
                    # Arrived at base location
                    if self.environment.base and world_to_grid(self.pos.x, self.pos.y) == self.environment.base.grid_pos:
                         print(f"Agent {self.id}: Arrived at base for needs.")
                         # Prioritize eating if hungry, otherwise rest
                         if self.hunger >= HIGH_HUNGER_THRESHOLD:
                             self.state = STATE_EATING
                             self.action_timer = EAT_TIME
                             print(f"Agent {self.id}: Starting to eat.")
                         elif self.energy <= LOW_ENERGY_THRESHOLD:
                             self.state = STATE_RESTING
                             self.action_timer = REST_TIME
                             print(f"Agent {self.id}: Starting to rest.")
                         else:
                              # Needs met by the time agent arrived? Go idle.
                              self.state = STATE_IDLE
                    else:
                         # Path finished but not at base?
                         print(f"Agent {self.id}: Finished path to base but not at base location.")
                         self.state = STATE_IDLE # Go idle and re-evaluate


            elif self.state == STATE_EATING:
                self.action_timer -= dt
                if self.action_timer <= 0:
                    self.hunger -= EAT_AMOUNT
                    self.hunger = max(0, self.hunger)
                    print(f"Agent {self.id}: Finished eating. Hunger: {self.hunger:.0f}")
                    # Check if still needs rest, otherwise idle
                    if self.energy <= LOW_ENERGY_THRESHOLD:
                        self.state = STATE_RESTING
                        self.action_timer = REST_TIME
                    else:
                        self.state = STATE_IDLE

            elif self.state == STATE_RESTING:
                 self.action_timer -= dt
                 if self.action_timer <= 0:
                     self.energy += REST_AMOUNT
                     self.energy = min(MAX_ENERGY, self.energy)
                     print(f"Agent {self.id}: Finished resting. Energy: {self.energy:.0f}")
                     # Check if still needs eating (unlikely but possible), otherwise idle
                     if self.hunger >= HIGH_HUNGER_THRESHOLD:
                         self.state = STATE_EATING
                         self.action_timer = EAT_TIME
                     else:
                         self.state = STATE_IDLE


    def _find_nearest_available_resource(self):
        """Finds the closest resource with quantity > 0 that isn't targeted."""
        min_dist_sq = float('inf') # Use squared distance for comparison efficiency
        nearest = None
        for resource in self.environment.get_available_resources():
            if resource.quantity > 0:
                # Use grid distance (heuristic) as a quick filter? Or just world distance?
                # Let's use world distance for now, pathfinding cost is the real factor later.
                dx = self.pos.x - resource.pos.x
                dy = self.pos.y - resource.pos.y
                d_sq = dx*dx + dy*dy
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    nearest = resource
        return nearest

    def draw(self, screen):
        """Draws the agent on the screen."""
        # Base color
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), AGENT_SIZE)

        # State Indicator (inner color)
        state_color = WHITE
        if self.state == STATE_MOVING_TO_RESOURCE: state_color = CYAN
        elif self.state == STATE_MOVING_TO_BASE: state_color = PURPLE
        elif self.state == STATE_COLLECTING: state_color = ORANGE
        elif self.state == STATE_RETURNING: state_color = YELLOW
        elif self.state == STATE_EATING: state_color = RED
        elif self.state == STATE_RESTING: state_color = GREEN
        pygame.draw.circle(screen, state_color, (int(self.pos.x), int(self.pos.y)), AGENT_SIZE // 2)

        # Draw path (optional, for debugging)
        # if self.current_path:
        #     path_points = [self.pos] + [grid_to_world_center(node[0], node[1]) for node in self.current_path[self.path_index:]]
        #     if len(path_points) > 1:
        #          pygame.draw.lines(screen, self.color, False, path_points, 1)

        # Draw Needs Bars (optional)
        # Energy Bar (Green)
        energy_ratio = self.energy / MAX_ENERGY
        pygame.draw.rect(screen, GREEN, (self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE - 6, AGENT_SIZE * 2 * energy_ratio, 3))
        # Hunger Bar (Red) - Higher value is worse, so draw inverse
        hunger_ratio = self.hunger / MAX_HUNGER
        pygame.draw.rect(screen, RED, (self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE - 3, AGENT_SIZE * 2 * hunger_ratio, 3))


class Environment:
    """Manages the simulation space, agents, resources, obstacles, grid, and base."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.resources = []
        self.obstacles = []
        self.base = None
        self.targeted_resources = {} # resource_id -> agent_id targeting it
        self.total_base_resources = 0

        # Initialize grid
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)] # 0 = walkable, 1 = obstacle

    def add_agent(self, agent):
        """Adds an agent to the environment."""
        self.agents.append(agent)

    def remove_agent(self, agent_to_remove):
        """Removes an agent from the environment."""
        # Release any resource the agent might have been targeting
        if agent_to_remove.target_resource:
            self.mark_resource_available(agent_to_remove.target_resource)

        # Remove from agent list
        self.agents = [agent for agent in self.agents if agent.id != agent_to_remove.id]
        print(f"Agent {agent_to_remove.id} removed. Remaining agents: {len(self.agents)}")


    def add_resource(self, resource):
        """Adds a resource and ensures its grid cell is walkable."""
        gx, gy = world_to_grid(resource.pos.x, resource.pos.y)
        if self.grid[gy][gx] == 0: # Only add if cell is walkable
            self.resources.append(resource)
            resource.grid_pos = (gx, gy) # Store grid pos
        else:
            print(f"Warning: Tried to add resource at obstacle location ({gx}, {gy}). Resource not added.")

    def add_obstacle(self, obstacle):
        """Adds an obstacle and updates the grid."""
        gx, gy = obstacle.grid_pos
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            if self.grid[gy][gx] == 0: # Check if not already an obstacle
                 # Check if base or resource exists there - prevent adding obstacle on top
                 if self.base and self.base.grid_pos == (gx, gy):
                     print(f"Warning: Cannot place obstacle on Base at ({gx}, {gy}).")
                     return
                 for res in self.resources:
                     if res.grid_pos == (gx, gy):
                         print(f"Warning: Cannot place obstacle on Resource {res.id} at ({gx}, {gy}).")
                         return

                 self.obstacles.append(obstacle)
                 self.grid[gy][gx] = 1 # Mark as obstacle
            # else: print(f"Warning: Obstacle already exists at ({gx}, {gy}).") # Optional warning
        else:
             print(f"Warning: Obstacle position ({gx}, {gy}) is out of grid bounds.")


    def set_base(self, base):
        """Sets the base location and ensures its grid cell is walkable."""
        gx, gy = world_to_grid(base.pos.x, base.pos.y)
        if self.grid[gy][gx] == 0: # Only add if cell is walkable
            self.base = base
            base.grid_pos = (gx, gy)
        else:
             print(f"Warning: Tried to set base at obstacle location ({gx}, {gy}). Base not set.")
             self.base = None # Ensure base is None if placement failed

    def add_base_resources(self, amount):
        self.total_base_resources += amount

    def create_random_resource(self):
        """Creates a resource at a random walkable grid cell."""
        attempts = 0
        while attempts < 100: # Prevent infinite loop if grid is full
            gx = random.randint(0, GRID_WIDTH - 1)
            gy = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[gy][gx] == 0: # Check if walkable
                 # Also check if it's the base location
                 if not (self.base and self.base.grid_pos == (gx, gy)):
                     wx, wy = grid_to_world_center(gx, gy)
                     self.add_resource(Resource(wx, wy))
                     return # Success
            attempts += 1
        print("Warning: Could not find a walkable spot for random resource after 100 attempts.")


    def create_random_agent(self):
        """Creates an agent at a random walkable grid cell."""
        attempts = 0
        while attempts < 100:
            gx = random.randint(0, GRID_WIDTH - 1)
            gy = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[gy][gx] == 0: # Check if walkable
                # Also check if it's the base location
                 if not (self.base and self.base.grid_pos == (gx, gy)):
                     wx, wy = grid_to_world_center(gx, gy)
                     self.add_agent(Agent(wx, wy, self))
                     return # Success
            attempts += 1
        print("Warning: Could not find a walkable spot for random agent after 100 attempts.")

    def create_obstacle_line(self, x1, y1, x2, y2):
         """Creates a line of obstacles on the grid (Bresenham's line algorithm)."""
         gx1, gy1 = world_to_grid(x1, y1)
         gx2, gy2 = world_to_grid(x2, y2)

         dx = abs(gx2 - gx1)
         dy = -abs(gy2 - gy1)
         sx = 1 if gx1 < gx2 else -1
         sy = 1 if gy1 < gy2 else -1
         err = dx + dy  # error value e_xy

         while True:
             self.add_obstacle(Obstacle(gx1, gy1))
             if gx1 == gx2 and gy1 == gy2:
                 break
             e2 = 2 * err
             if e2 >= dy:  # e_xy+e_x > 0
                 err += dy
                 gx1 += sx
             if e2 <= dx:  # e_xy+e_y < 0
                 err += dx
                 gy1 += sy

    def mark_resource_targeted(self, resource, agent):
        """Marks a resource as being targeted by an agent."""
        if resource and resource.id not in self.targeted_resources:
            self.targeted_resources[resource.id] = agent.id
            # print(f"Resource {resource.id} targeted by Agent {agent.id}")

    def mark_resource_available(self, resource):
        """Marks a resource as available again."""
        if resource and resource.id in self.targeted_resources:
            del self.targeted_resources[resource.id]
            # print(f"Resource {resource.id} is now available")

    def is_resource_targeted_by(self, resource, agent):
        """Checks if a resource is specifically targeted by the given agent."""
        return self.targeted_resources.get(resource.id) == agent.id

    def get_available_resources(self):
        """Returns a list of resources not currently targeted."""
        return [res for res in self.resources if res.id not in self.targeted_resources]

    def handle_collisions(self, dt):
        """Simple collision handling: push overlapping agents apart."""
        # Using spatial hashing or quadtrees would be better for large numbers
        agents_list = self.agents
        agent_count = len(agents_list)
        push_force = AGENT_SPEED * 1.5 # How strongly to push apart

        for i in range(agent_count):
            agent1 = agents_list[i]
            for j in range(i + 1, agent_count):
                agent2 = agents_list[j]

                dist_vec = agent1.pos - agent2.pos
                dist_len = dist_vec.length()
                min_dist = AGENT_SIZE # Collision if distance is less than sum of radii (AGENT_SIZE/2 * 2)

                if dist_len < min_dist and dist_len > 0: # Check dist_len > 0 to avoid division by zero if positions are identical
                    overlap = min_dist - dist_len
                    push_vec = dist_vec.normalize() * overlap * 0.5 # Push each agent half the overlap distance

                    # Apply push - directly modify position slightly
                    # We multiply by push_force * dt to make it somewhat frame-rate independent, though this is approximate
                    move1 = push_vec * push_force * dt
                    move2 = -push_vec * push_force * dt

                    # Check potential movement against obstacles before applying
                    next_pos1 = agent1.pos + move1
                    next_pos2 = agent2.pos + move2
                    grid_x1, grid_y1 = world_to_grid(next_pos1.x, next_pos1.y)
                    grid_x2, grid_y2 = world_to_grid(next_pos2.x, next_pos2.y)

                    if self.grid[grid_y1][grid_x1] == 0:
                         agent1.pos += move1
                    if self.grid[grid_y2][grid_x2] == 0:
                         agent2.pos += move2

                    # Boundary check after push
                    agent1.pos.x = max(AGENT_SIZE//2, min(SCREEN_WIDTH - AGENT_SIZE//2, agent1.pos.x))
                    agent1.pos.y = max(AGENT_SIZE//2, min(SCREEN_HEIGHT - AGENT_SIZE//2, agent1.pos.y))
                    agent2.pos.x = max(AGENT_SIZE//2, min(SCREEN_WIDTH - AGENT_SIZE//2, agent2.pos.x))
                    agent2.pos.y = max(AGENT_SIZE//2, min(SCREEN_HEIGHT - AGENT_SIZE//2, agent2.pos.y))



    def update(self, dt):
        """Updates all agents and resources in the environment."""
        # Update resources (regeneration)
        for resource in self.resources:
            resource.update(dt)

        # Update agents
        # Iterate over a copy in case agents are removed during update (e.g., dying)
        for agent in self.agents[:]:
            agent.update(dt)

        # Handle collisions after movement updates
        self.handle_collisions(dt)


    def draw_grid(self, screen):
        """Draws the grid lines."""
        for x in range(0, SCREEN_WIDTH, GRID_CELL_SIZE):
            pygame.draw.line(screen, GREY, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, GRID_CELL_SIZE):
            pygame.draw.line(screen, GREY, (0, y), (SCREEN_WIDTH, y), 1)

    def draw(self, screen, draw_grid_flag):
        """Draws the environment contents."""
        if draw_grid_flag:
            self.draw_grid(screen)

        # Draw base
        if self.base:
            self.base.draw(screen)

        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        # Draw resources
        for resource in self.resources:
            is_targeted = resource.id in self.targeted_resources
            resource.draw(screen, is_targeted)

        # Draw agents
        for agent in self.agents:
            agent.draw(screen)

# --- Main Game Loop ---
def main():
    pygame.init()
    pygame.font.init() # Initialize font system
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Complex Real-Time Agent Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24) # Font for UI text

    # --- Create Environment and Entities ---
    environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Create Base (ensure it's not on an edge for pathfinding simplicity)
    base_x = SCREEN_WIDTH // 2
    base_y = SCREEN_HEIGHT // 2
    environment.set_base(Base(base_x, base_y))
    if not environment.base: # Handle case where base placement failed (e.g. obstacle pre-placed)
        print("FATAL: Could not place base. Exiting.")
        return

    # Create some obstacles
    # Make a border?
    # for x in range(GRID_WIDTH):
    #     environment.add_obstacle(Obstacle(x, 0))
    #     environment.add_obstacle(Obstacle(x, GRID_HEIGHT - 1))
    # for y in range(1, GRID_HEIGHT - 1):
    #      environment.add_obstacle(Obstacle(0, y))
    #      environment.add_obstacle(Obstacle(GRID_WIDTH - 1, y))
    # Random lines
    for _ in range(5):
         x1 = random.randint(50, SCREEN_WIDTH - 50)
         y1 = random.randint(50, SCREEN_HEIGHT - 50)
         x2 = x1 + random.randint(-200, 200)
         y2 = y1 + random.randint(-200, 200)
         environment.create_obstacle_line(x1, y1, x2, y2)


    # Create initial resources (placed randomly on walkable cells)
    for _ in range(15):
        environment.create_random_resource()

    # Create initial agents (placed randomly on walkable cells)
    for _ in range(20):
        environment.create_random_agent()

    running = True
    draw_grid_flag = False # Toggle grid visibility
    paused = False

    while running:
        # --- Delta Time Calculation ---
        dt = clock.tick(FPS) / 1000.0
        # Avoid huge dt spikes if debugging or paused
        if dt > 0.1: dt = 0.1
        if paused: dt = 0 # No time passes if paused

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_g: # Toggle grid
                    draw_grid_flag = not draw_grid_flag
                if event.key == pygame.K_p: # Pause
                    paused = not paused
                if event.key == pygame.K_a: # Add agent
                    environment.create_random_agent()
                if event.key == pygame.K_r: # Add resource
                    environment.create_random_resource()
                if event.key == pygame.K_o: # Add obstacle line (draw with mouse?) - Simple random line for now
                    x1 = random.randint(50, SCREEN_WIDTH - 50)
                    y1 = random.randint(50, SCREEN_HEIGHT - 50)
                    x2 = x1 + random.randint(-100, 100)
                    y2 = y1 + random.randint(-100, 100)
                    environment.create_obstacle_line(x1, y1, x2, y2)


        # --- Updates (only if not paused) ---
        if not paused:
            environment.update(dt)

        # --- Drawing ---
        screen.fill(BLACK)         # Clear screen
        environment.draw(screen, draw_grid_flag)   # Draw environment contents

        # Draw UI Text
        agent_count_text = font.render(f"Agents: {len(environment.agents)}", True, WHITE)
        resource_count_text = font.render(f"Resources: {len(environment.resources)}", True, WHITE)
        base_storage_text = font.render(f"Base Storage: {environment.total_base_resources}", True, WHITE)
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, WHITE)
        pause_text = font.render("PAUSED", True, YELLOW)

        screen.blit(agent_count_text, (10, 10))
        screen.blit(resource_count_text, (10, 30))
        screen.blit(base_storage_text, (10, 50))
        screen.blit(fps_text, (SCREEN_WIDTH - 80, 10))
        if paused:
            screen.blit(pause_text, (SCREEN_WIDTH // 2 - pause_text.get_width() // 2, 10))


        pygame.display.flip()      # Update the display

    pygame.quit()

if __name__ == '__main__':
    main()