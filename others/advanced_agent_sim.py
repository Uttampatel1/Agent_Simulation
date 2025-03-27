import pygame
import random
import math
import time
import heapq  # For A* priority queue
import pickle # For Save/Load
import os     # For checking save file existence

# --- Constants ---
SCREEN_WIDTH = 1300 # Increased width for UI panel
SCREEN_HEIGHT = 800
UI_PANEL_WIDTH = 200 # Width of the right-side UI panel
SIM_WIDTH = SCREEN_WIDTH - UI_PANEL_WIDTH
SIM_HEIGHT = SCREEN_HEIGHT
GRID_CELL_SIZE = 20
GRID_WIDTH = SIM_WIDTH // GRID_CELL_SIZE
GRID_HEIGHT = SIM_HEIGHT // GRID_CELL_SIZE

AGENT_SIZE = 8 # Radius
AGENT_SPEED = 70  # Max speed Pixels per second
AGENT_MAX_FORCE = 180.0 # Max steering force
RESOURCE_SIZE = 15
OBSTACLE_COLOR = (80, 80, 80)
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
DARK_GREY = (50, 50, 50)
UI_BG_COLOR = (30, 30, 50) # Dark blue/purple for UI panel
BUTTON_COLOR = (80, 80, 110)
BUTTON_HOVER_COLOR = (110, 110, 140)
BUTTON_TEXT_COLOR = WHITE
SELECTED_BUTTON_COLOR = (150, 150, 180)

# Agent States / Action Types
STATE_IDLE = "IDLE"
STATE_MOVING_TO_RESOURCE = "MOVING_TO_RESOURCE"
STATE_MOVING_TO_BASE = "MOVING_TO_BASE"
STATE_MOVING_TO_BUILD = "MOVING_TO_BUILD"
STATE_WORKING = "WORKING" # Generic state for timed actions

# Action Types (used within STATE_WORKING)
ACTION_COLLECTING = "COLLECTING"
ACTION_RETURNING = "RETURNING"
ACTION_EATING = "EATING"
ACTION_RESTING = "RESTING"
ACTION_BUILDING = "BUILDING"

# Needs Constants
MAX_ENERGY = 100
MAX_HUNGER = 100
ENERGY_DECAY_RATE = 1.5
HUNGER_INCREASE_RATE = 2.0
LOW_ENERGY_THRESHOLD = 30
HIGH_HUNGER_THRESHOLD = 70
EAT_AMOUNT = 60
REST_AMOUNT = 70
EAT_TIME = 1.5
REST_TIME = 2.5

# Resource Constants
RESOURCE_MAX_QUANTITY = 50
RESOURCE_REGEN_RATE = 0.3

# Building Constants
BUILD_TIME = 3.0
BUILD_COST = 2

# Terrain Costs / Types
TERRAIN_EMPTY = 0 # Impassable (Obstacles, Walls) - A* cost
TERRAIN_PLAINS = 1
TERRAIN_GRASS = 2
TERRAIN_MUD = 5
TERRAIN_WALL = 0 # A* cost for walls

TERRAIN_TYPES = {
    "PLAINS": TERRAIN_PLAINS,
    "GRASS": TERRAIN_GRASS,
    "MUD": TERRAIN_MUD,
    "WALL": TERRAIN_WALL,
    "OBSTACLE": OBSTACLE_COLOR # Same as empty/wall for pathfinding
}

TERRAIN_COLORS = {
    TERRAIN_EMPTY: (50, 50, 50),
    TERRAIN_PLAINS: (210, 180, 140), # Tan
    TERRAIN_GRASS: (34, 139, 34),   # Forest Green
    TERRAIN_MUD: (139, 69, 19),     # Saddle Brown
    TERRAIN_WALL: OBSTACLE_COLOR,
    OBSTACLE_COLOR: OBSTACLE_COLOR
}
OBSTACLE_COST = TERRAIN_EMPTY # Ensure obstacle cost makes it impassable

# Quadtree Constants
QT_MAX_OBJECTS = 4
QT_MAX_LEVELS = 6

# UI Constants / Paint Brushes
UI_PANEL_X = SIM_WIDTH
UI_FONT_SIZE_SMALL = 18
UI_FONT_SIZE_MEDIUM = 20
UI_LINE_HEIGHT = 22
BUTTON_HEIGHT = 30
BUTTON_PADDING = 5

BRUSH_SELECT = "SELECT"
BRUSH_WALL = "WALL"
BRUSH_PLAINS = "PLAINS"
BRUSH_GRASS = "GRASS"
BRUSH_MUD = "MUD"
BRUSH_CLEAR = "CLEAR" # Essentially paints plains

PAINT_BRUSHES = [
    BRUSH_SELECT, BRUSH_WALL, BRUSH_CLEAR, BRUSH_PLAINS, BRUSH_GRASS, BRUSH_MUD
]

# --- Helper Functions ---
def world_to_grid(world_x, world_y):
    # Ensure coordinates are within sim area before converting
    if world_x < 0 or world_x >= SIM_WIDTH or world_y < 0 or world_y >= SIM_HEIGHT:
        return None # Indicate outside sim area
    grid_x = int(world_x // GRID_CELL_SIZE)
    grid_y = int(world_y // GRID_CELL_SIZE)
    # Clamp to grid bounds (should be less necessary with area check)
    grid_x = max(0, min(GRID_WIDTH - 1, grid_x))
    grid_y = max(0, min(GRID_HEIGHT - 1, grid_y))
    return grid_x, grid_y

def grid_to_world_center(grid_x, grid_y):
    world_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    world_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    return world_x, world_y

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Quadtree Class (Identical to previous version) ---
class Quadtree:
    """ Basic Quadtree for 2D spatial partitioning. Objects need a 'rect' attribute. """
    def __init__(self, level, bounds):
        self.level = level
        self.bounds = bounds
        self.objects = []
        self.nodes = [None, None, None, None] # NW, NE, SW, SE

    def clear(self):
        self.objects = []
        for i in range(len(self.nodes)):
            if self.nodes[i] is not None:
                self.nodes[i].clear()
        self.nodes = [None, None, None, None]

    def _split(self):
        sub_width = self.bounds.width / 2
        sub_height = self.bounds.height / 2
        x, y = self.bounds.x, self.bounds.y
        # Ensure children bounds are within parent (handle potential float issues)
        nw_bounds = pygame.Rect(x, y, math.ceil(sub_width), math.ceil(sub_height))
        ne_bounds = pygame.Rect(x + sub_width, y, math.floor(sub_width), math.ceil(sub_height))
        sw_bounds = pygame.Rect(x, y + sub_height, math.ceil(sub_width), math.floor(sub_height))
        se_bounds = pygame.Rect(x + sub_width, y + sub_height, math.floor(sub_width), math.floor(sub_height))

        self.nodes[0] = Quadtree(self.level + 1, nw_bounds)
        self.nodes[1] = Quadtree(self.level + 1, ne_bounds)
        self.nodes[2] = Quadtree(self.level + 1, sw_bounds)
        self.nodes[3] = Quadtree(self.level + 1, se_bounds)


    def _get_index(self, rect):
        index = -1
        if not self.bounds.contains(rect): # Quick check if rect is outside bounds
             return -1

        vert_mid = self.bounds.x + self.bounds.width / 2
        horz_mid = self.bounds.y + self.bounds.height / 2
        top_q = (rect.bottom <= horz_mid) # Use bottom/right for containment check
        bot_q = (rect.top >= horz_mid)
        left_q = (rect.right <= vert_mid)
        right_q = (rect.left >= vert_mid)

        if left_q:
            if top_q: index = 0 # NW
            elif bot_q: index = 2 # SW
        elif right_q:
            if top_q: index = 1 # NE
            elif bot_q: index = 3 # SE
        return index

    def insert(self, obj):
        if not hasattr(obj, 'rect'): return
        # Do not insert if object is outside this node's bounds
        if not self.bounds.colliderect(obj.rect):
            return

        if self.nodes[0] is not None:
            index = self._get_index(obj.rect)
            if index != -1:
                self.nodes[index].insert(obj)
                return
        self.objects.append(obj)
        if len(self.objects) > QT_MAX_OBJECTS and self.level < QT_MAX_LEVELS:
            if self.nodes[0] is None: self._split()
            i = 0
            while i < len(self.objects):
                index = self._get_index(self.objects[i].rect)
                if index != -1:
                    self.nodes[index].insert(self.objects.pop(i))
                else: i += 1

    def query(self, query_rect):
        found = []
        # Check against objects in this node that intersect
        found.extend([obj for obj in self.objects if query_rect.colliderect(obj.rect)])

        if self.nodes[0] is not None:
            # Check children that intersect the query rect
            for i in range(4):
                if self.nodes[i].bounds.colliderect(query_rect):
                    found.extend(self.nodes[i].query(query_rect))
        return found

    def query_radius(self, center_pos, radius):
        radius_sq = radius * radius
        query_bounds = pygame.Rect(center_pos[0] - radius, center_pos[1] - radius, radius * 2, radius * 2)
        # Ensure query bounds don't go outside main simulation area for Quadtree
        query_bounds.clamp_ip(pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT))

        potential = self.query(query_bounds)
        # Precise circle check
        nearby = [obj for obj in potential
                  if hasattr(obj, 'pos') and obj.pos.distance_squared_to(center_pos) <= radius_sq]
        return nearby

    def draw(self, screen):
         pygame.draw.rect(screen, DARK_GREY, self.bounds, 1)
         if self.nodes[0] is not None:
             for node in self.nodes: node.draw(screen)

# --- A* Pathfinding Function (Identical to previous version) ---
def astar_pathfinding(grid, start_node, end_node):
    """ Finds path using A* considering terrain costs. grid stores costs (0=impassable)."""
    if not grid or not (0 <= start_node[0] < len(grid[0])) or \
       not (0 <= start_node[1] < len(grid)) or \
       not (0 <= end_node[0] < len(grid[0])) or \
       not (0 <= end_node[1] < len(grid)) or \
       grid[start_node[1]][start_node[0]] == OBSTACLE_COST or \
       grid[end_node[1]][end_node[0]] == OBSTACLE_COST:
        return None

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4-way movement

    close_set = set()
    came_from = {}
    gscore = {start_node: 0}
    fscore = {start_node: heuristic(start_node, end_node)}
    oheap = []
    heapq.heappush(oheap, (fscore[start_node], start_node))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if not (0 <= neighbor[0] < len(grid[0]) and 0 <= neighbor[1] < len(grid)):
                continue

            neighbor_cost = grid[neighbor[1]][neighbor[0]]
            if neighbor_cost == OBSTACLE_COST or neighbor in close_set:
                continue

            tentative_g_score = gscore[current] + neighbor_cost

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end_node)
                # Optimization: Check if neighbor is already in heap with higher fscore?
                # heapq doesn't support decrease-key easily, adding duplicates is common.
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None


# --- Entity Classes (Obstacle, Base, Resource identical to previous version) ---
class Obstacle:
    def __init__(self, grid_x, grid_y):
        self.grid_pos = (grid_x, grid_y)
        self.rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        self.color = OBSTACLE_COLOR

    def draw(self, screen):
        # Only draw if within sim area (though should be placed there)
        if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Base:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.rect = pygame.Rect(x - BASE_SIZE // 2, y - BASE_SIZE // 2, BASE_SIZE, BASE_SIZE)
        self.grid_pos = world_to_grid(x, y)
        self.color = BASE_COLOR

    def draw(self, screen):
         if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Resource:
    _id_counter = 0
    def __init__(self, x, y):
        self.id = Resource._id_counter
        Resource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = world_to_grid(x, y)
        # Ensure grid_pos is valid before proceeding
        if self.grid_pos is None:
            raise ValueError(f"Resource created outside sim bounds: ({x},{y})")
        self.quantity = random.randint(RESOURCE_MAX_QUANTITY // 2, RESOURCE_MAX_QUANTITY)
        self.max_quantity = RESOURCE_MAX_QUANTITY
        self.regen_timer = 0
        self.rect = pygame.Rect(self.pos.x - RESOURCE_SIZE // 2, self.pos.y - RESOURCE_SIZE // 2, RESOURCE_SIZE, RESOURCE_SIZE)

    def collect(self, amount=1):
        collected = min(amount, self.quantity)
        self.quantity -= collected
        self.quantity = max(0, self.quantity)
        return collected

    def update(self, dt):
        if self.quantity < self.max_quantity:
            self.regen_timer += dt
            regen_interval = 1.0 / RESOURCE_REGEN_RATE if RESOURCE_REGEN_RATE > 0 else float('inf')
            if self.regen_timer >= regen_interval:
                self.quantity = min(self.max_quantity, self.quantity + 1)
                self.regen_timer -= regen_interval

    def draw(self, screen, is_targeted):
        if self.rect.right > SIM_WIDTH or self.rect.bottom > SIM_HEIGHT: return # Don't draw outside sim

        if is_targeted: color = YELLOW
        elif self.quantity == 0: color = DARK_GREY
        else:
            ratio = self.quantity / self.max_quantity
            red_val = int(255 * (1 - ratio))
            green_val = int(255 * ratio)
            color = (red_val, green_val, 0)
        pygame.draw.rect(screen, color, self.rect)


# --- Agent Class (Mostly same, minor adjustments for sim area) ---
class Agent:
    _id_counter = 0
    def __init__(self, x, y, environment):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.environment = environment
        self.pos = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
        self.max_speed = AGENT_SPEED
        self.max_force = AGENT_MAX_FORCE
        self.color = BLUE
        self.state = STATE_IDLE
        self.action_type = None
        self.action_timer = 0
        self.target_resource = None
        self.target_pos_world = None
        self.destination_grid_pos = None
        self.current_path = []
        self.path_index = 0
        self.energy = MAX_ENERGY * random.uniform(0.8, 1.0)
        self.hunger = MAX_HUNGER * random.uniform(0.0, 0.2)
        self.carrying_resource = 0
        # Clamp initial position just in case
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))
        self.rect = pygame.Rect(self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE, AGENT_SIZE * 2, AGENT_SIZE * 2)


    def update_rect(self):
        self.rect.center = (int(self.pos.x), int(self.pos.y))

    def apply_force(self, force):
        self.velocity += force

    def seek(self, target_pos):
        if target_pos is None: return pygame.Vector2(0, 0)
        desired = (target_pos - self.pos)
        dist_sq = desired.length_squared()
        if dist_sq < 4: return pygame.Vector2(0, 0) # Threshold slightly larger

        desired.normalize_ip()
        desired *= self.max_speed
        steer = desired - self.velocity
        if steer.length_squared() > self.max_force * self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def separation(self, neighbors):
        steer = pygame.Vector2(0, 0)
        count = 0
        desired_separation = (AGENT_SIZE * 2) * 1.5

        for other in neighbors:
            if other is self: continue
            dist_sq = self.pos.distance_squared_to(other.pos)
            if 0 < dist_sq < desired_separation * desired_separation:
                diff = self.pos - other.pos
                # Weight by inverse distance squared (stronger repulsion closer)
                # Add small epsilon to avoid division by zero
                diff *= (1.0 / (dist_sq + 0.001))
                steer += diff
                count += 1

        if count > 0:
            steer /= count
            if steer.length_squared() > 0:
                # Desired separation velocity = steer normalized * max_speed
                steer.normalize_ip()
                steer *= self.max_speed
                # Calculate steering force = desired - current velocity
                steer -= self.velocity
                # Limit the steering force
                if steer.length_squared() > self.max_force * self.max_force:
                    steer.scale_to_length(self.max_force)
        return steer

    def _find_path(self, target_grid_pos):
        """ Sets agent's path using A* if possible. """
        start_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        self.current_path = []
        self.path_index = 0
        self.target_pos_world = None
        self.destination_grid_pos = target_grid_pos # Store final target

        if start_grid_pos is None or target_grid_pos is None: return False # Cannot path if outside grid
        if start_grid_pos == target_grid_pos: return True # Already there

        path = astar_pathfinding(self.environment.grid, start_grid_pos, target_grid_pos)

        if path:
            self.current_path = path
            self.path_index = 0
            next_grid_node = self.current_path[self.path_index]
            self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            return True
        else:
            self.destination_grid_pos = None # Clear destination if path failed
            return False

    def update(self, dt, nearby_agents):
        """ Main update logic: Needs, State Machine, Steering, Movement. """
        # --- Needs Update ---
        self.energy -= ENERGY_DECAY_RATE * dt
        self.hunger += HUNGER_INCREASE_RATE * dt
        self.energy = max(0, self.energy)
        self.hunger = min(MAX_HUNGER, self.hunger)

        if self.energy <= 0:
            # print(f"Agent {self.id}: Died of exhaustion!") # Can be noisy
            self.environment.remove_agent(self)
            return

        # --- Steering Forces Calculation ---
        seek_force = pygame.Vector2(0, 0)
        # Separation weight can be adjusted
        separation_force = self.separation(nearby_agents) * 1.5

        # --- High Priority Need Checks & State Overrides ---
        needs_override = False
        is_moving_for_needs = self.state == STATE_MOVING_TO_BASE and (self.action_type == ACTION_EATING or self.action_type == ACTION_RESTING)

        # Check hunger only if not already eating/moving to eat
        if self.hunger >= HIGH_HUNGER_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             if self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 if self._find_path(self.environment.base.grid_pos):
                     # print(f"Agent {self.id}: Critically hungry! Seeking base.") # Debug
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_EATING
                     self._release_resource_target()
                     needs_override = True
                 # else: print(f"Agent {self.id}: Hungry, cannot path to base!") # Debug

        # Check energy only if not already resting/moving to rest and not handling hunger
        elif self.energy <= LOW_ENERGY_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             if not needs_override and self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 if self._find_path(self.environment.base.grid_pos):
                     # print(f"Agent {self.id}: Critically tired! Seeking base.") # Debug
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_RESTING
                     self._release_resource_target()
                     needs_override = True
                 # else: print(f"Agent {self.id}: Tired, cannot path to base!") # Debug

        # --- State Machine & Path Following Logic ---
        path_completed_this_frame = False

        # Path following
        if self.current_path:
            if self.target_pos_world:
                 dist_to_node_sq = self.pos.distance_squared_to(self.target_pos_world)
                 arrival_threshold_sq = (GRID_CELL_SIZE * 0.5)**2 # Slightly larger arrival radius

                 if dist_to_node_sq < arrival_threshold_sq:
                     self.path_index += 1
                     if self.path_index < len(self.current_path):
                         next_node_grid = self.current_path[self.path_index]
                         self.target_pos_world = pygame.Vector2(grid_to_world_center(next_node_grid[0], next_node_grid[1]))
                     else: # Reached end of path
                         self.current_path = []
                         self.target_pos_world = None
                         path_completed_this_frame = True
                         if self.destination_grid_pos: # Snap to final grid center
                             self.pos = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))
                         self.velocity *= 0.1 # Damp velocity on arrival
            else: # Path exists but no target node? Clear path.
                self.current_path = []
                self.path_index = 0

        # Apply seek force towards the current path node or final destination if path ended
        seek_target = self.target_pos_world
        if path_completed_this_frame and self.destination_grid_pos:
             seek_target = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))

        if seek_target:
             seek_force = self.seek(seek_target)

        # --- State Logic ---
        if not needs_override:
            # Idle state: Decide next action
            if self.state == STATE_IDLE:
                if self.carrying_resource > 0: # Return resources
                    if self.environment.base and self._find_path(self.environment.base.grid_pos):
                         self.state = STATE_MOVING_TO_BASE
                         self.action_type = ACTION_RETURNING
                # Build? (Less frequent)
                elif self.carrying_resource >= BUILD_COST and self.energy > LOW_ENERGY_THRESHOLD + 20 and random.random() < 0.01:
                    build_spot = self._find_build_spot()
                    if build_spot and self._find_path(build_spot):
                         self.state = STATE_MOVING_TO_BUILD
                else: # Find resource
                    resource_target = self._find_best_available_resource()
                    if resource_target and self._find_path(resource_target.grid_pos):
                        self.state = STATE_MOVING_TO_RESOURCE
                        self.target_resource = resource_target
                        self.environment.mark_resource_targeted(resource_target, self)

            # Moving states: Transition upon path completion
            elif self.state == STATE_MOVING_TO_RESOURCE and path_completed_this_frame:
                if self.target_resource and self.target_resource.quantity > 0 and self.environment.is_resource_targeted_by(self.target_resource, self):
                    self.state = STATE_WORKING
                    self.action_type = ACTION_COLLECTING
                    # Dynamic collect time based on quantity? Small variation.
                    self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                else: # Resource gone or taken
                    self._release_resource_target()
                    self.state = STATE_IDLE

            elif self.state == STATE_MOVING_TO_BUILD and path_completed_this_frame:
                current_grid_pos = world_to_grid(self.pos.x, self.pos.y)
                build_target = self.destination_grid_pos
                # Check if arrived at intended build spot and it's still valid
                if build_target and current_grid_pos == build_target and \
                   self.environment.is_buildable(build_target[0], build_target[1]):
                    self.state = STATE_WORKING
                    self.action_type = ACTION_BUILDING
                    self.action_timer = BUILD_TIME
                    # Keep destination_grid_pos to know where to build
                else: # Arrived but target invalid or changed?
                    self.destination_grid_pos = None
                    self.state = STATE_IDLE

            elif self.state == STATE_MOVING_TO_BASE and path_completed_this_frame:
                 if self.action_type == ACTION_RETURNING:
                     self.environment.add_base_resources(self.carrying_resource)
                     self.carrying_resource = 0
                     self.action_type = None
                     self.state = STATE_IDLE # Re-evaluate needs/tasks
                 elif self.action_type == ACTION_EATING:
                     self.state = STATE_WORKING
                     self.action_timer = EAT_TIME
                 elif self.action_type == ACTION_RESTING:
                     self.state = STATE_WORKING
                     self.action_timer = REST_TIME
                 else: # Arrived at base for unknown reason
                     self.action_type = None
                     self.state = STATE_IDLE

            # Working state: Perform timed action
            elif self.state == STATE_WORKING:
                self.action_timer -= dt
                if self.action_timer <= 0:
                    action_done = False
                    # Perform action based on type
                    if self.action_type == ACTION_COLLECTING:
                        if self.target_resource and self.target_resource.quantity > 0:
                            collected = self.target_resource.collect(1)
                            self.carrying_resource += collected
                            # Decide: Continue or stop?
                            if self.target_resource.quantity == 0 or self.carrying_resource >= 5: # Simple capacity
                                self._release_resource_target()
                                action_done = True
                            else: # Reset timer to collect more
                                self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                        else: # Resource depleted while collecting
                            self._release_resource_target()
                            action_done = True

                    elif self.action_type == ACTION_BUILDING:
                         build_pos = self.destination_grid_pos
                         if build_pos and self.carrying_resource >= BUILD_COST:
                              build_success = self.environment.build_wall(build_pos[0], build_pos[1], self)
                              if build_success: self.carrying_resource -= BUILD_COST
                         # else: Failed or not enough resources
                         self.destination_grid_pos = None # Clear build target regardless
                         action_done = True

                    elif self.action_type == ACTION_EATING:
                         self.hunger = max(0, self.hunger - EAT_AMOUNT)
                         action_done = True
                    elif self.action_type == ACTION_RESTING:
                         self.energy = min(MAX_ENERGY, self.energy + REST_AMOUNT)
                         action_done = True
                    else: # Unknown action
                         action_done = True

                    # If action finished, go idle
                    if action_done:
                         self.action_type = None
                         self.state = STATE_IDLE

        # --- Apply Forces & Update Movement ---
        total_force = seek_force + separation_force
        # Apply force (consider dt for frame-rate independence?)
        # Accel = total_force / mass (assume mass=1) => Accel = total_force
        # Velocity change = Accel * dt = total_force * dt
        self.velocity += total_force * dt # Apply force scaled by dt


        # Limit velocity
        vel_mag_sq = self.velocity.length_squared()
        if vel_mag_sq > self.max_speed * self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        # Damp velocity significantly if working or idle without a steering target
        elif self.state == STATE_WORKING:
             self.velocity *= 0.05 # Almost stop
        elif self.state == STATE_IDLE and not self.target_pos_world and not self.current_path:
             self.velocity *= 0.85 # Slow down gradually

        # Update position
        self.pos += self.velocity * dt

        # --- Boundary Constraints (within SIMULATION AREA) ---
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))

        # Update Rect for Quadtree and drawing
        self.update_rect()


    def _release_resource_target(self):
        if self.target_resource:
            self.environment.mark_resource_available(self.target_resource)
            self.target_resource = None

    def _find_best_available_resource(self, max_search_radius=300):
        best_score = -float('inf')
        best_resource = None
        # Query quadtree for potentially relevant resources
        potential_resources = self.environment.quadtree.query_radius(self.pos, max_search_radius)
        available_resources = [
            res for res in potential_resources
            if isinstance(res, Resource) and res.quantity > 0 and res.id not in self.environment.targeted_resources
        ]
        if not available_resources: return None

        weight_quantity = 1.5
        weight_distance = -0.05

        for resource in available_resources:
            dist_sq = self.pos.distance_squared_to(resource.pos)
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1
            score = (weight_quantity * resource.quantity) + (weight_distance * dist)
            if score > best_score:
                 # Optimization: Maybe skip path check here for performance? Assume targetable if found.
                 # Check if path exists (can be expensive if done often)
                 # if astar_pathfinding(self.environment.grid, world_to_grid(*self.pos), resource.grid_pos):
                      best_score = score
                      best_resource = resource
                 # else: Resource inaccessible

        return best_resource


    def _find_build_spot(self):
        my_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        if my_grid_pos is None: return None # Agent outside grid?

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            check_x, check_y = my_grid_pos[0] + dx, my_grid_pos[1] + dy
            if self.environment.is_buildable(check_x, check_y):
                 return (check_x, check_y)
        return None

    def draw(self, screen):
         # Don't draw if outside simulation area
         if self.pos.x >= SIM_WIDTH or self.pos.y >= SIM_HEIGHT: return

         # Base color circle
         pygame.draw.circle(screen, self.color, self.rect.center, AGENT_SIZE)
         # State Indicator
         state_color = WHITE
         current_state = self.state
         current_action = self.action_type
         # ... (State color logic - same as before) ...
         if current_state == STATE_MOVING_TO_RESOURCE: state_color = CYAN
         elif current_state == STATE_MOVING_TO_BUILD: state_color = ORANGE
         elif current_state == STATE_MOVING_TO_BASE:
             if current_action == ACTION_RETURNING: state_color = YELLOW
             elif current_action == ACTION_EATING: state_color = RED
             elif current_action == ACTION_RESTING: state_color = GREEN
             else: state_color = PURPLE
         elif current_state == STATE_WORKING:
             if current_action == ACTION_COLLECTING: state_color = ORANGE
             elif current_action == ACTION_BUILDING: state_color = GREY
             elif current_action == ACTION_EATING: state_color = RED
             elif current_action == ACTION_RESTING: state_color = GREEN
             else: state_color = DARK_GREY
         pygame.draw.circle(screen, state_color, self.rect.center, max(1, AGENT_SIZE // 2)) # Inner circle

         # Needs Bars below agent
         bar_width = AGENT_SIZE * 2
         bar_x = self.pos.x - bar_width / 2
         bar_y_energy = self.pos.y + AGENT_SIZE + 2
         bar_y_hunger = bar_y_energy + 4

         energy_ratio = self.energy / MAX_ENERGY
         pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_energy, bar_width, 3))
         pygame.draw.rect(screen, GREEN, (bar_x, bar_y_energy, bar_width * energy_ratio, 3))

         hunger_ratio = self.hunger / MAX_HUNGER
         pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_hunger, bar_width, 3))
         pygame.draw.rect(screen, RED, (bar_x, bar_y_hunger, bar_width * hunger_ratio, 3))

# --- UI Button Class ---
class Button:
    def __init__(self, x, y, w, h, text, action, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action # String identifier for the action
        self.font = font
        self.is_hovered = False
        self.is_selected = False # For toggle buttons like paint brushes

    def draw(self, screen):
        color = BUTTON_COLOR
        if self.is_selected:
            color = SELECTED_BUTTON_COLOR
        elif self.is_hovered:
            color = BUTTON_HOVER_COLOR

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 1) # Border

        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_click(self):
        if self.is_hovered:
            # print(f"Button Clicked: {self.action}") # Debug
            return self.action
        return None

# --- Environment Class ---
class Environment:
    def __init__(self, width, height, sim_width, sim_height, font_small, font_medium):
        self.width = width
        self.height = height
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.font_small = font_small
        self.font_medium = font_medium
        self.agents = []
        self.resources = []
        self.obstacles = []
        self.base = None
        self.total_base_resources = 0
        self.targeted_resources = {}

        self.grid = [[TERRAIN_PLAINS for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.quadtree = Quadtree(0, pygame.Rect(0, 0, sim_width, sim_height))

        # UI State
        self.selected_agent = None
        self.speed_multiplier = 1.0
        self.paint_brush = BRUSH_SELECT
        self.buttons = []
        self._setup_ui_buttons()


    def _setup_ui_buttons(self):
        """ Creates the UI buttons """
        self.buttons = []
        button_font = self.font_small
        bw = (UI_PANEL_WIDTH - 3 * BUTTON_PADDING) // 2 # Button width (2 columns)
        bh = BUTTON_HEIGHT
        bx = UI_PANEL_X + BUTTON_PADDING
        by = 10 # Starting Y

        # --- Control Buttons ---
        self.buttons.append(Button(bx, by, bw, bh, "Pause", "toggle_pause", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Play", "play", button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, bw, bh, "Slow (-)", "speed_down", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Fast (+)", "speed_up", button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, bw*2 + BUTTON_PADDING, bh, "Reset Speed", "speed_reset", button_font))
        by += bh + BUTTON_PADDING * 3

        # --- Toggle Buttons ---
        self.buttons.append(Button(bx, by, bw, bh, "Grid", "toggle_grid", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Quadtree", "toggle_quadtree", button_font))
        by += bh + BUTTON_PADDING * 3

        # --- Paint Brush Buttons ---
        brush_bw = UI_PANEL_WIDTH - 2 * BUTTON_PADDING # Full width for brushes
        self.buttons.append(Button(bx, by, brush_bw, bh, "Select Tool", ("set_brush", BRUSH_SELECT), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Wall", ("set_brush", BRUSH_WALL), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Clear", ("set_brush", BRUSH_CLEAR), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Plains", ("set_brush", BRUSH_PLAINS), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Grass", ("set_brush", BRUSH_GRASS), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Mud", ("set_brush", BRUSH_MUD), button_font))
        by += bh + BUTTON_PADDING * 3

         # --- Save/Load Buttons ---
        self.buttons.append(Button(bx, by, bw, bh, "Save (F5)", "save", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Load (F9)", "load", button_font))

        # Update selected state for the initial brush
        self._update_button_selected_state()

    def _update_button_selected_state(self):
        """ Updates the is_selected flag for buttons based on current state. """
        for button in self.buttons:
            if isinstance(button.action, tuple) and button.action[0] == "set_brush":
                button.is_selected = (button.action[1] == self.paint_brush)
            # Could add similar logic for pause/play state if desired

    def handle_ui_click(self, mouse_pos, game_state):
        """ Checks if a UI button was clicked and performs its action. """
        clicked_action = None
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                clicked_action = button.handle_click()
                break # Only handle one button click

        if clicked_action:
            if clicked_action == "toggle_pause": game_state['paused'] = not game_state['paused']
            elif clicked_action == "play": game_state['paused'] = False
            elif clicked_action == "speed_down": self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
            elif clicked_action == "speed_up": self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5)
            elif clicked_action == "speed_reset": self.speed_multiplier = 1.0
            elif clicked_action == "toggle_grid": game_state['draw_grid'] = not game_state['draw_grid']
            elif clicked_action == "toggle_quadtree": game_state['draw_quadtree'] = not game_state['draw_quadtree']
            elif clicked_action == "save": return "save" # Signal to main loop
            elif clicked_action == "load": return "load" # Signal to main loop
            elif isinstance(clicked_action, tuple) and clicked_action[0] == "set_brush":
                self.paint_brush = clicked_action[1]
                self._update_button_selected_state()
            return True # Indicate UI handled the click
        return False

    def paint_terrain(self, mouse_pos, brush_size=1):
        """ Paints terrain/walls onto the grid based on current brush. """
        grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
        if grid_pos is None: return # Outside sim area

        gx, gy = grid_pos
        terrain_type_to_paint = None
        is_wall_or_obstacle = False

        if self.paint_brush == BRUSH_WALL:
            terrain_type_to_paint = TERRAIN_WALL # Use cost directly
            is_wall_or_obstacle = True
        elif self.paint_brush == BRUSH_CLEAR:
             terrain_type_to_paint = TERRAIN_PLAINS # Clear sets to plains
        elif self.paint_brush == BRUSH_PLAINS:
             terrain_type_to_paint = TERRAIN_PLAINS
        elif self.paint_brush == BRUSH_GRASS:
              terrain_type_to_paint = TERRAIN_GRASS
        elif self.paint_brush == BRUSH_MUD:
               terrain_type_to_paint = TERRAIN_MUD

        if terrain_type_to_paint is not None:
             # Apply paint in a square area based on brush_size
             offset = brush_size // 2
             for dx in range(-offset, offset + 1):
                 for dy in range(-offset, offset + 1):
                      paint_x, paint_y = gx + dx, gy + dy
                      if 0 <= paint_x < GRID_WIDTH and 0 <= paint_y < GRID_HEIGHT:
                           # Check if painting over base or resource
                           is_base = self.base and self.base.grid_pos == (paint_x, paint_y)
                           is_res = any(r.grid_pos == (paint_x, paint_y) for r in self.resources)
                           if not is_base and not is_res:
                               self.grid[paint_y][paint_x] = terrain_type_to_paint
                               # If painting wall/obstacle, add to obstacle list? Less critical now.
                               # If clearing wall/obstacle, remove from list? Might be complex.
                               # For simplicity, rely on grid cost mainly.

    def is_buildable(self, grid_x, grid_y):
         """ Checks if a grid cell is suitable for building a wall on. """
         if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             terrain_cost = self.grid[grid_y][grid_x]
             # Buildable on plains or grass, not mud, not existing wall/obstacle
             if terrain_cost != OBSTACLE_COST and terrain_cost != TERRAIN_WALL and terrain_cost != TERRAIN_MUD:
                  # Check not base or resource spot
                  if not (self.base and self.base.grid_pos == (grid_x, grid_y)):
                      is_res_spot = any(r.grid_pos == (grid_x, grid_y) for r in self.resources)
                      if not is_res_spot:
                          return True
         return False

    # --- Add/Remove/Update methods (mostly same as before) ---
    def add_agent(self, agent):
        self.agents.append(agent)
        # Quadtree rebuild handles insertion

    def remove_agent(self, agent_to_remove):
        agent_to_remove._release_resource_target()
        self.agents = [a for a in self.agents if a.id != agent_to_remove.id]
        if self.selected_agent and self.selected_agent.id == agent_to_remove.id:
             self.selected_agent = None

    def add_resource(self, resource):
        if resource.grid_pos is None: return # Skip if created outside grid
        gx, gy = resource.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.resources.append(resource)
        # else: Cannot place resource on obstacle

    def add_obstacle(self, obstacle):
        gx, gy = obstacle.grid_pos
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
             if self.grid[gy][gx] != OBSTACLE_COST:
                  if self.base and self.base.grid_pos == (gx, gy): return
                  if any(r.grid_pos == (gx, gy) for r in self.resources): return
                  # self.obstacles.append(obstacle) # Less needed if grid is master
                  self.grid[gy][gx] = OBSTACLE_COST

    def set_base(self, base):
        if base.grid_pos is None: return # Cannot place base outside grid
        gx, gy = base.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.base = base
        else: print(f"Warning: Cannot set base at occupied grid cell {base.grid_pos}")

    def add_base_resources(self, amount): self.total_base_resources += amount
    def build_wall(self, grid_x, grid_y, builder_agent): # Renamed from add_wall
         if self.is_buildable(grid_x, grid_y):
             self.grid[grid_y][grid_x] = TERRAIN_WALL
             return True
         return False

    def create_random_entity(self, entity_type):
        attempts = 0
        while attempts < 100:
            # Create within sim area bounds
            gx = random.randint(1, GRID_WIDTH - 2)
            gy = random.randint(1, GRID_HEIGHT - 2)
            # Check if buildable (more general than just walkable)
            if self.is_buildable(gx, gy):
                 wx, wy = grid_to_world_center(gx, gy)
                 try:
                     if entity_type == 'agent': self.add_agent(Agent(wx, wy, self))
                     elif entity_type == 'resource': self.add_resource(Resource(wx, wy))
                     return
                 except ValueError as e: # Catch errors like resource outside bounds
                      print(f"Error creating entity: {e}")
                      # Continue trying
            attempts += 1
        print(f"Warning: Could not find spot for random {entity_type}.")

    def create_obstacle_line(self, x1, y1, x2, y2):
        # Clamp coords to sim area first
        x1 = max(0, min(SIM_WIDTH - 1, x1)); y1 = max(0, min(SIM_HEIGHT - 1, y1))
        x2 = max(0, min(SIM_WIDTH - 1, x2)); y2 = max(0, min(SIM_HEIGHT - 1, y2))
        # Convert potentially clamped coords to grid
        start_node = world_to_grid(x1, y1)
        end_node = world_to_grid(x2, y2)
        if start_node is None or end_node is None: return # Outside grid

        gx1, gy1 = start_node; gx2, gy2 = end_node
        dx = abs(gx2 - gx1); dy = -abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1; sy = 1 if gy1 < gy2 else -1
        err = dx + dy
        while True:
             self.add_obstacle(Obstacle(gx1, gy1)) # Internally checks validity
             if gx1 == gx2 and gy1 == gy2: break
             e2 = 2 * err
             if e2 >= dy: err += dy; gx1 += sx
             if e2 <= dx: err += dx; gy1 += sy

    def mark_resource_targeted(self, resource, agent):
        if resource and resource.id not in self.targeted_resources:
            self.targeted_resources[resource.id] = agent.id
    def mark_resource_available(self, resource):
        if resource and resource.id in self.targeted_resources:
            del self.targeted_resources[resource.id]
    def is_resource_targeted_by(self, resource, agent):
        return self.targeted_resources.get(resource.id) == agent.id

    def select_agent_at(self, mouse_pos):
        # Ensure selection only within sim area
        if mouse_pos[0] >= self.sim_width or mouse_pos[1] >= self.sim_height:
             self.selected_agent = None
             return

        self.selected_agent = None
        search_radius = AGENT_SIZE * 2.5
        # Use Quadtree query centered on mouse pos
        nearby_entities = self.quadtree.query_radius(mouse_pos, search_radius)
        nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]

        min_dist_sq = search_radius * search_radius
        closest_agent = None
        for agent in nearby_agents:
             dist_sq = agent.pos.distance_squared_to(mouse_pos)
             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_agent = agent
        self.selected_agent = closest_agent
        # if self.selected_agent: print(f"Selected Agent {self.selected_agent.id}") # Debug

    def update_agents(self, dt):
         agents_to_update = self.agents[:]
         for agent in agents_to_update:
              if agent in self.agents:
                   nearby_radius = (AGENT_SIZE * 2) * 3
                   nearby_entities = self.quadtree.query_radius(agent.pos, nearby_radius)
                   nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]
                   agent.update(dt * self.speed_multiplier, nearby_agents) # Apply speed multiplier here

    def update_resources(self, dt):
        for resource in self.resources:
            resource.update(dt * self.speed_multiplier) # Apply speed multiplier

    def rebuild_quadtree(self):
        self.quadtree.clear()
        # Insert agents that are within the simulation bounds
        for agent in self.agents:
            if 0 <= agent.pos.x < self.sim_width and 0 <= agent.pos.y < self.sim_height:
                agent.update_rect()
                self.quadtree.insert(agent)
        # Insert resources that are within the simulation bounds
        for resource in self.resources:
             if 0 <= resource.pos.x < self.sim_width and 0 <= resource.pos.y < self.sim_height:
                if not hasattr(resource, 'rect'): # Ensure rect exists
                    resource.rect = pygame.Rect(0,0,RESOURCE_SIZE, RESOURCE_SIZE)
                resource.rect.center = resource.pos
                self.quadtree.insert(resource)

    # --- Drawing Methods ---
    def draw_sim_area(self, screen, draw_grid_flag, draw_quadtree_flag):
        """ Draws only the simulation area contents. """
        # Optional: Fill background for sim area only? Or let grid draw handle it.
        # sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
        # screen.fill(BLACK, sim_rect) # Fill if needed

        if draw_grid_flag: self.draw_grid(screen)
        if self.base: self.base.draw(screen)
        # for obs in self.obstacles: obs.draw(screen) # Less needed if grid draws walls
        for res in self.resources: res.draw(screen, res.id in self.targeted_resources)
        for agent in self.agents: agent.draw(screen) # Agent draw checks bounds
        if draw_quadtree_flag: self.quadtree.draw(screen)

        # Highlight selected agent
        if self.selected_agent and self.selected_agent in self.agents:
             # Ensure highlight stays within sim bounds
             center_x = int(self.selected_agent.pos.x)
             center_y = int(self.selected_agent.pos.y)
             if center_x < self.sim_width and center_y < self.sim_height:
                 pygame.draw.circle(screen, WHITE, (center_x, center_y), AGENT_SIZE + 3, 1)

    def draw_grid(self, screen):
        """ Draws grid cells with terrain colors within sim area. """
        sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
        # Optimization: Only draw visible grid cells? For now, draw all.
        for gy in range(GRID_HEIGHT):
            for gx in range(GRID_WIDTH):
                terrain_cost = self.grid[gy][gx]
                color = TERRAIN_COLORS.get(terrain_cost, GREY)
                rect = pygame.Rect(gx * GRID_CELL_SIZE, gy * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
                # Only draw if rect is within the screen's simulation area
                if sim_rect.contains(rect): # Check if fully contained for simplicity
                    pygame.draw.rect(screen, color, rect)
                    # Optional grid lines
                    # pygame.draw.rect(screen, DARK_GREY, rect, 1)

    def draw_ui(self, screen, clock, game_state):
        """ Draws the UI panel on the right. """
        ui_panel_rect = pygame.Rect(UI_PANEL_X, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, UI_BG_COLOR, ui_panel_rect)
        pygame.draw.line(screen, WHITE, (UI_PANEL_X, 0), (UI_PANEL_X, SCREEN_HEIGHT), 1) # Separator

        # --- Draw Buttons ---
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)
            button.draw(screen)

        # --- Draw Info Text ---
        info_y = 380 # Starting Y for info text below buttons
        line = 0
        def draw_info(text, val=""):
             nonlocal line
             full_text = f"{text}: {val}" if val else text
             surf = self.font_small.render(full_text, True, WHITE)
             screen.blit(surf, (UI_PANEL_X + BUTTON_PADDING, info_y + line * UI_LINE_HEIGHT))
             line += 1

        draw_info("FPS", f"{int(clock.get_fps())}")
        draw_info("Speed", f"{self.speed_multiplier:.1f}x")
        draw_info("Agents", f"{len(self.agents)}")
        draw_info("Resources", f"{len(self.resources)}")
        draw_info("Base Storage", f"{self.total_base_resources}")
        draw_info("Brush", f"{self.paint_brush}")
        line +=1 # Add spacing

        # --- Selected Agent Info ---
        draw_info("Selected Agent:")
        if self.selected_agent and self.selected_agent in self.agents:
            agent = self.selected_agent
            draw_info("  ID", agent.id)
            state_str = agent.state
            if agent.state == STATE_WORKING and agent.action_type:
                 state_str += f" ({agent.action_type})"
            draw_info("  State", state_str)
            draw_info("  Energy", f"{agent.energy:.1f}")
            draw_info("  Hunger", f"{agent.hunger:.1f}")
            draw_info("  Carrying", agent.carrying_resource)
            if agent.target_resource:
                 draw_info("  Target", f"Res {agent.target_resource.id}")
            elif agent.destination_grid_pos:
                 draw_info("  Target", f"Grid {agent.destination_grid_pos}")

        else:
             draw_info("  None")

        # Draw Pause indicator if paused
        if game_state['paused']:
             pause_surf = self.font_medium.render("PAUSED", True, YELLOW)
             pause_rect = pause_surf.get_rect(centerx=UI_PANEL_X + UI_PANEL_WIDTH // 2, y=SCREEN_HEIGHT - 40)
             screen.blit(pause_surf, pause_rect)


# --- Save/Load Functions ---
SAVE_FILENAME = "agent_sim_save_v2.pkl"

def save_simulation(environment, filename=SAVE_FILENAME):
    font_small_backup = environment.font_small
    font_medium_backup = environment.font_medium
    selected_id_backup = environment.selected_agent.id if environment.selected_agent else None
    buttons_backup = environment.buttons # Buttons might contain font refs
    environment.font_small = None
    environment.font_medium = None
    environment.selected_agent = None
    environment.buttons = [] # Don't save buttons (recreated on load)
    environment.quadtree = None # Don't save quadtree

    save_data = {
        'environment_state': environment.__dict__, # Save attributes dict
        'selected_agent_id': selected_id_backup
    }

    try:
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Simulation saved to {filename}")
    except Exception as e:
        print(f"Error saving simulation: {e}")
    finally:
        # Restore transient objects immediately (though they might be overwritten on load)
        environment.font_small = font_small_backup
        environment.font_medium = font_medium_backup
        environment.buttons = buttons_backup


def load_simulation(filename=SAVE_FILENAME):
    if not os.path.exists(filename):
        print(f"Save file '{filename}' not found.")
        return None, None

    try:
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # Recreate environment instance and load state
        # Pass dummy fonts initially, they'll be overwritten
        loaded_env = Environment(
            save_data['environment_state']['width'],
            save_data['environment_state']['height'],
            save_data['environment_state']['sim_width'],
            save_data['environment_state']['sim_height'],
            None, None
        )
        loaded_env.__dict__.update(save_data['environment_state'])
        selected_id = save_data['selected_agent_id']

        print(f"Simulation loaded from {filename}")
        return loaded_env, selected_id
    except Exception as e:
        print(f"Error loading simulation: {e}")
        return None, None


# --- Main Game Loop ---
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Agent Simulation")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont(None, UI_FONT_SIZE_SMALL)
    font_medium = pygame.font.SysFont(None, UI_FONT_SIZE_MEDIUM)

    # --- Game State Dictionary ---
    game_state = {
        'paused': False,
        'draw_grid': True,
        'draw_quadtree': False,
        'is_painting': False # Track if mouse button is down for painting
    }

    # --- Create Environment ---
    environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT, SIM_WIDTH, SIM_HEIGHT, font_small, font_medium)

    # --- Initial Population ---
    base_x, base_y = SIM_WIDTH // 3, SIM_HEIGHT // 2
    environment.set_base(Base(base_x, base_y))
    if not environment.base: print("Warning: Failed to place base."); return

    # Make base area walkable initially
    if environment.base.grid_pos:
         gx, gy = environment.base.grid_pos
         for dx in [-1, 0, 1]:
             for dy in [-1, 0, 1]:
                  if 0 <= gx+dx < GRID_WIDTH and 0 <= gy+dy < GRID_HEIGHT:
                      environment.grid[gy+dy][gx+dx] = TERRAIN_PLAINS

    # Add Obstacles
    environment.create_obstacle_line(SIM_WIDTH * 0.6, SIM_HEIGHT * 0.1, SIM_WIDTH * 0.7, SIM_HEIGHT * 0.9)
    environment.create_obstacle_line(SIM_WIDTH * 0.1, SIM_HEIGHT * 0.8, SIM_WIDTH * 0.5, SIM_HEIGHT * 0.7)

    # Add Resources & Agents
    for _ in range(40): environment.create_random_entity('resource')
    for _ in range(60): environment.create_random_entity('agent')

    # --- Main Loop ---
    running = True
    while running:
        # --- Delta Time ---
        base_dt = clock.tick(FPS) / 1000.0
        dt = min(base_dt, 0.1) # Clamp dt
        effective_dt = dt if not game_state['paused'] else 0 # Actual dt used for updates

        # --- Event Handling ---
        mouse_pos = pygame.mouse.get_pos()
        sim_area_rect = pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT)
        mouse_in_sim = sim_area_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # --- Keyboard Input ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                # Use buttons now for most controls, keep keys for save/load?
                elif event.key == pygame.K_F5:
                     save_simulation(environment)
                     # Need to restore transient state after saving attempt
                     environment.font_small = font_small
                     environment.font_medium = font_medium
                     environment._setup_ui_buttons() # Recreate buttons with font refs
                     environment._update_button_selected_state()
                     if hasattr(environment, '__dict__'): # Check if load wiped env prematurely
                        environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))

                elif event.key == pygame.K_F9:
                    loaded_env, selected_id = load_simulation()
                    if loaded_env:
                        environment = loaded_env
                        # Restore transient state after loading
                        environment.font_small = font_small
                        environment.font_medium = font_medium
                        environment._setup_ui_buttons() # Recreate buttons
                        environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
                        # Find and set selected agent
                        environment.selected_agent = None
                        if selected_id is not None:
                             for agent in environment.agents:
                                 if agent.id == selected_id:
                                     environment.selected_agent = agent
                                     break
                        print("Simulation Loaded.")
                # Add agent/resource keys?
                elif event.key == pygame.K_a: environment.create_random_entity('agent')
                elif event.key == pygame.K_r: environment.create_random_entity('resource')

            # --- Mouse Input ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    # Check UI Panel First
                    if mouse_pos[0] >= UI_PANEL_X:
                        action = environment.handle_ui_click(mouse_pos, game_state)
                        if action == "save":
                            save_simulation(environment)
                            # Restore transient state
                            environment.font_small = font_small
                            environment.font_medium = font_medium
                            environment._setup_ui_buttons()
                            environment._update_button_selected_state()
                            if hasattr(environment, '__dict__'):
                                environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
                        elif action == "load":
                            # Load logic handled via F9 key for simplicity now
                             print("Load via F9")
                             pass


                    # If not UI, check Sim Area
                    elif mouse_in_sim:
                        if environment.paint_brush == BRUSH_SELECT:
                            environment.select_agent_at(mouse_pos)
                        else: # Start painting
                            game_state['is_painting'] = True
                            environment.paint_terrain(mouse_pos) # Paint single cell on click

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1: # Left Release
                      game_state['is_painting'] = False

            elif event.type == pygame.MOUSEMOTION:
                 if game_state['is_painting'] and mouse_in_sim:
                      if environment.paint_brush != BRUSH_SELECT:
                           environment.paint_terrain(mouse_pos) # Paint on drag

        # --- Updates (if not paused) ---
        if not game_state['paused']:
            # Rebuild Quadtree (essential for dynamic objects)
            environment.rebuild_quadtree()
            # Update Resources
            environment.update_resources(dt) # dt already includes speed multiplier in method
            # Update Agents
            environment.update_agents(dt) # dt already includes speed multiplier in method

        # --- Drawing ---
        screen.fill(BLACK) # Clear screen
        # Draw Simulation Area
        environment.draw_sim_area(screen, game_state['draw_grid'], game_state['draw_quadtree'])
        # Draw UI Panel
        environment.draw_ui(screen, clock, game_state)

        pygame.display.flip() # Update the full display

    pygame.quit()

# --- Run ---
if __name__ == '__main__':
    main()