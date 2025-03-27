
import pygame
import random
import math
import time
import heapq  # For A* priority queue
import pickle # For Save/Load
import os     # For checking save file existence

# --- Constants ---
# Screen and layout dimensions
SCREEN_WIDTH = 1300 # Increased width to accommodate the UI panel
SCREEN_HEIGHT = 800
UI_PANEL_WIDTH = 200 # Width of the right-side user interface panel
SIM_WIDTH = SCREEN_WIDTH - UI_PANEL_WIDTH # Width of the simulation area
SIM_HEIGHT = SCREEN_HEIGHT             # Height of the simulation area

# Grid configuration for pathfinding and terrain
GRID_CELL_SIZE = 20
GRID_WIDTH = SIM_WIDTH // GRID_CELL_SIZE
GRID_HEIGHT = SIM_HEIGHT // GRID_CELL_SIZE

# Agent properties
AGENT_SIZE = 8 # Radius of the agent circle
AGENT_SPEED = 70  # Maximum speed in pixels per second
AGENT_MAX_FORCE = 180.0 # Maximum steering force applied per second

# Other entity properties
RESOURCE_SIZE = 15 # Width/Height of resource squares
OBSTACLE_COLOR = (80, 80, 80) # Color for obstacles/walls
BASE_SIZE = 40     # Width/Height of the base square
BASE_COLOR = (200, 200, 0) # Color for the base

# Simulation performance
FPS = 60 # Target frames per second

# Standard Colors
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

# UI specific colors
UI_BG_COLOR = (30, 30, 50) # Background color for the UI panel
BUTTON_COLOR = (80, 80, 110) # Default button color
BUTTON_HOVER_COLOR = (110, 110, 140) # Button color when hovered
BUTTON_TEXT_COLOR = WHITE
SELECTED_BUTTON_COLOR = (150, 150, 180) # Color for selected/active toggle buttons

# Agent States / Action Types used for behavior logic
STATE_IDLE = "IDLE" # Agent is waiting for a task
STATE_MOVING_TO_RESOURCE = "MOVING_TO_RESOURCE" # Agent is pathing towards a resource
STATE_MOVING_TO_BASE = "MOVING_TO_BASE" # Agent is pathing towards the base (for various reasons)
STATE_MOVING_TO_BUILD = "MOVING_TO_BUILD" # Agent is pathing to a build location
STATE_WORKING = "WORKING" # Agent is performing a timed action (e.g., collecting, eating)

# Action Types (used within STATE_WORKING to specify the action)
ACTION_COLLECTING = "COLLECTING"
ACTION_RETURNING = "RETURNING" # Technically done when moving, but set as intention for MOVING_TO_BASE
ACTION_EATING = "EATING"
ACTION_RESTING = "RESTING"
ACTION_BUILDING = "BUILDING"

# Agent Needs Constants
MAX_ENERGY = 100
MAX_HUNGER = 100
ENERGY_DECAY_RATE = 1.5 # Energy lost per second
HUNGER_INCREASE_RATE = 2.0 # Hunger gained per second
LOW_ENERGY_THRESHOLD = 30 # Threshold below which agent seeks rest
HIGH_HUNGER_THRESHOLD = 70 # Threshold above which agent seeks food
EAT_AMOUNT = 60    # Hunger reduced when eating
REST_AMOUNT = 70   # Energy restored when resting
EAT_TIME = 1.5     # Seconds to complete eating action
REST_TIME = 2.5    # Seconds to complete resting action

# Resource Constants
RESOURCE_MAX_QUANTITY = 50 # Max resources a node can hold
RESOURCE_REGEN_RATE = 0.3  # Resources regenerated per second

# Building Constants
BUILD_TIME = 3.0 # Seconds to build a wall segment
BUILD_COST = 2   # Resources required to build a wall segment

# Terrain Costs / Types for pathfinding and drawing
# Cost 0 means impassable for A*
TERRAIN_EMPTY = 0 # Impassable (Represents Obstacles/Walls in the cost grid)
TERRAIN_PLAINS = 1 # Low movement cost
TERRAIN_GRASS = 2  # Slightly higher cost
TERRAIN_MUD = 5    # High movement cost
TERRAIN_WALL = 0 # Impassable, same cost as TERRAIN_EMPTY

# Dictionary mapping terrain names to their costs
TERRAIN_TYPES = {
    "PLAINS": TERRAIN_PLAINS,
    "GRASS": TERRAIN_GRASS,
    "MUD": TERRAIN_MUD,
    "WALL": TERRAIN_WALL,
    "OBSTACLE": OBSTACLE_COLOR # Used for identifying obstacles, cost is TERRAIN_EMPTY/WALL
}

# Dictionary mapping terrain costs to their drawing colors
TERRAIN_COLORS = {
    TERRAIN_EMPTY: (50, 50, 50),   # Dark grey for impassable areas
    TERRAIN_PLAINS: (210, 180, 140), # Tan color
    TERRAIN_GRASS: (34, 139, 34),    # Forest Green
    TERRAIN_MUD: (139, 69, 19),      # Saddle Brown
    TERRAIN_WALL: OBSTACLE_COLOR,    # Same as obstacle color
    OBSTACLE_COLOR: OBSTACLE_COLOR   # Mapping obstacle color constant itself
}
OBSTACLE_COST = TERRAIN_EMPTY # Ensure obstacle cost makes it impassable in A*

# Quadtree Constants for spatial partitioning performance
QT_MAX_OBJECTS = 4 # Max objects in a node before splitting
QT_MAX_LEVELS = 6  # Max depth of the quadtree

# UI Constants / Paint Brush Identifiers
UI_PANEL_X = SIM_WIDTH # X-coordinate where the UI panel starts
UI_FONT_SIZE_SMALL = 18
UI_FONT_SIZE_MEDIUM = 20
UI_LINE_HEIGHT = 22 # Spacing for text lines in UI
BUTTON_HEIGHT = 30
BUTTON_PADDING = 5

# Identifiers for the terrain painting brushes
BRUSH_SELECT = "SELECT" # Tool for selecting agents
BRUSH_WALL = "WALL"     # Tool for painting walls/obstacles
BRUSH_PLAINS = "PLAINS" # Tool for painting plains terrain
BRUSH_GRASS = "GRASS"   # Tool for painting grass terrain
BRUSH_MUD = "MUD"     # Tool for painting mud terrain
BRUSH_CLEAR = "CLEAR"   # Tool to clear walls/obstacles (resets to PLAINS)

# List of available paint brushes for UI creation
PAINT_BRUSHES = [
    BRUSH_SELECT, BRUSH_WALL, BRUSH_CLEAR, BRUSH_PLAINS, BRUSH_GRASS, BRUSH_MUD
]

# --- Helper Functions ---
def world_to_grid(world_x, world_y):
    """Converts world pixel coordinates to grid cell indices."""
    # Check if the coordinates are outside the simulation area
    if world_x < 0 or world_x >= SIM_WIDTH or world_y < 0 or world_y >= SIM_HEIGHT:
        return None # Indicate coordinates are outside the valid simulation grid area
    grid_x = int(world_x // GRID_CELL_SIZE)
    grid_y = int(world_y // GRID_CELL_SIZE)
    # Clamp to ensure grid indices stay within bounds (redundant with initial check, but safe)
    grid_x = max(0, min(GRID_WIDTH - 1, grid_x))
    grid_y = max(0, min(GRID_HEIGHT - 1, grid_y))
    return grid_x, grid_y

def grid_to_world_center(grid_x, grid_y):
    """Converts grid cell indices to the center world pixel coordinates of the cell."""
    world_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    world_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
    return world_x, world_y

def heuristic(a, b):
    """Calculates the Manhattan distance heuristic between two grid points (a, b)."""
    # Used by A* pathfinding.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Quadtree Class ---
class Quadtree:
    """
    A Quadtree implementation for 2D spatial partitioning.
    Objects inserted must have a 'rect' attribute (pygame.Rect).
    Helps optimize collision detection and finding nearby objects.
    """
    def __init__(self, level, bounds):
        """
        Initializes the Quadtree node.
        Args:
            level (int): The current depth level of this node.
            bounds (pygame.Rect): The rectangular boundary of this node.
        """
        self.level = level
        self.bounds = bounds
        self.objects = [] # Objects stored directly in this node
        self.nodes = [None, None, None, None] # Child nodes (NW, NE, SW, SE)

    def clear(self):
        """Removes all objects and children from this node and its descendants."""
        self.objects = []
        for i in range(len(self.nodes)):
            if self.nodes[i] is not None:
                self.nodes[i].clear()
                self.nodes[i] = None # Explicitly remove child reference

    def _split(self):
        """Divides this node into four equal child quadrants."""
        sub_width = self.bounds.width / 2
        sub_height = self.bounds.height / 2
        x, y = self.bounds.x, self.bounds.y

        # Define bounds for the four children, using ceil/floor to avoid gaps/overlaps
        nw_bounds = pygame.Rect(x, y, math.ceil(sub_width), math.ceil(sub_height))
        ne_bounds = pygame.Rect(x + sub_width, y, math.floor(sub_width), math.ceil(sub_height))
        sw_bounds = pygame.Rect(x, y + sub_height, math.ceil(sub_width), math.floor(sub_height))
        se_bounds = pygame.Rect(x + sub_width, y + sub_height, math.floor(sub_width), math.floor(sub_height))

        # Create child Quadtree nodes
        self.nodes[0] = Quadtree(self.level + 1, nw_bounds)
        self.nodes[1] = Quadtree(self.level + 1, ne_bounds)
        self.nodes[2] = Quadtree(self.level + 1, sw_bounds)
        self.nodes[3] = Quadtree(self.level + 1, se_bounds)


    def _get_index(self, rect):
        """
        Determines which quadrant a given rectangle belongs to within this node.
        Returns -1 if the rectangle cannot completely fit within a single quadrant
        or is outside the node's bounds.
        """
        index = -1
        # If the rect isn't fully contained, it stays in the parent node (or -1 if outside)
        if not self.bounds.contains(rect):
             return -1

        # Calculate midpoints
        vert_mid = self.bounds.x + self.bounds.width / 2
        horz_mid = self.bounds.y + self.bounds.height / 2

        # Check which quadrant(s) the rect overlaps
        top_q = (rect.bottom <= horz_mid) # Check if entirely in top half
        bot_q = (rect.top >= horz_mid)    # Check if entirely in bottom half
        left_q = (rect.right <= vert_mid) # Check if entirely in left half
        right_q = (rect.left >= vert_mid)   # Check if entirely in right half

        # Assign index based on quadrant containment
        if left_q:
            if top_q: index = 0 # North-West
            elif bot_q: index = 2 # South-West
        elif right_q:
            if top_q: index = 1 # North-East
            elif bot_q: index = 3 # South-East

        # If it spans multiple quadrants (didn't fit cleanly into one), index remains -1
        return index

    def insert(self, obj):
        """
        Inserts an object into the Quadtree.
        Args:
            obj: The object to insert. Must have a 'rect' attribute.
        """
        # Ignore objects without a rect attribute
        if not hasattr(obj, 'rect'): return
        # Do not insert if object's rect doesn't intersect this node's bounds at all
        if not self.bounds.colliderect(obj.rect):
            return

        # If this node has children, try to insert into the appropriate child
        if self.nodes[0] is not None:
            index = self._get_index(obj.rect)
            if index != -1: # If the object fits entirely in one child quadrant
                self.nodes[index].insert(obj)
                return

        # If no children or object spans multiple children, add to this node's list
        self.objects.append(obj)

        # If node exceeds capacity and hasn't reached max level, split it
        if len(self.objects) > QT_MAX_OBJECTS and self.level < QT_MAX_LEVELS:
            # Split only if not already split
            if self.nodes[0] is None:
                self._split()

            # Move objects from this node down to children if they fit
            i = 0
            while i < len(self.objects):
                index = self._get_index(self.objects[i].rect)
                if index != -1: # If object fits in a child
                    self.nodes[index].insert(self.objects.pop(i)) # Move object to child
                else:
                    i += 1 # Object stays here, check next one

    def query(self, query_rect):
        """
        Finds all objects in the Quadtree that intersect with a given rectangle.
        Args:
            query_rect (pygame.Rect): The rectangular area to query.
        Returns:
            list: A list of objects intersecting the query_rect.
        """
        found = []
        # Check objects in the current node
        found.extend([obj for obj in self.objects if query_rect.colliderect(obj.rect)])

        # If this node has children, query relevant children
        if self.nodes[0] is not None:
            # Check which children intersect the query rectangle
            for i in range(4):
                if self.nodes[i].bounds.colliderect(query_rect):
                    found.extend(self.nodes[i].query(query_rect)) # Recursively query child
        return found

    def query_radius(self, center_pos, radius):
        """
        Finds all objects within a given radius of a center point.
        Uses query() with a bounding box first, then does precise distance checks.
        Args:
            center_pos (tuple or pygame.Vector2): The center point (x, y).
            radius (float): The radius of the circular query area.
        Returns:
            list: A list of objects within the specified radius.
        """
        radius_sq = radius * radius
        # Create a bounding box around the circle for initial quadtree query
        query_bounds = pygame.Rect(center_pos[0] - radius, center_pos[1] - radius, radius * 2, radius * 2)
        # Ensure the query bounds stay within the simulation area (important for quadtree)
        query_bounds.clamp_ip(pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT))

        # Get potential candidates using the rectangular query
        potential = self.query(query_bounds)

        # Perform precise circular check on potential candidates
        nearby = []
        center_vec = pygame.Vector2(center_pos) # Ensure center is a Vector2 for distance check
        for obj in potential:
            # Check if object has a position and is within the radius
            if hasattr(obj, 'pos') and isinstance(obj.pos, pygame.Vector2):
                 if obj.pos.distance_squared_to(center_vec) <= radius_sq:
                     nearby.append(obj)
            # Fallback: Check based on rect center if pos is missing/wrong type
            elif hasattr(obj, 'rect'):
                 obj_center = pygame.Vector2(obj.rect.center)
                 if obj_center.distance_squared_to(center_vec) <= radius_sq:
                     nearby.append(obj)

        return nearby

    def draw(self, screen):
        """Draws the boundaries of this node and its children (for debugging)."""
        # Draw current node boundary
        pygame.draw.rect(screen, DARK_GREY, self.bounds, 1)
        # Recursively draw children if they exist
        if self.nodes[0] is not None:
            for node in self.nodes:
                node.draw(screen)

# --- A* Pathfinding Function ---
def astar_pathfinding(grid, start_node, end_node):
    """
    Finds the shortest path between two nodes on a grid using the A* algorithm.
    Considers terrain costs defined in the grid.

    Args:
        grid (list[list[int]]): 2D list representing the grid, where each cell contains its movement cost (0 = impassable).
        start_node (tuple[int, int]): The starting grid coordinates (x, y).
        end_node (tuple[int, int]): The target grid coordinates (x, y).

    Returns:
        list[tuple[int, int]] or None: A list of grid coordinates representing the path
                                       from start to end (exclusive of start), or None if no path exists.
    """
    # Basic validation checks
    if not grid or not start_node or not end_node: return None # Grid or nodes invalid
    rows, cols = len(grid), len(grid[0])
    if not (0 <= start_node[0] < cols and 0 <= start_node[1] < rows and \
            0 <= end_node[0] < cols and 0 <= end_node[1] < rows):
        return None # Start or end node out of bounds

    # Check if start or end node is impassable (cost is 0 or OBSTACLE_COST)
    if grid[start_node[1]][start_node[0]] == OBSTACLE_COST or \
       grid[end_node[1]][end_node[0]] == OBSTACLE_COST:
        return None

    # Define possible movements (4-directional)
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Down, Up, Right, Left

    close_set = set() # Set of nodes already evaluated
    came_from = {}    # Dictionary to reconstruct path: came_from[node] = previous_node
    gscore = {start_node: 0} # Cost from start along best known path
    fscore = {start_node: heuristic(start_node, end_node)} # Estimated total cost from start to end through node
    oheap = [] # Priority queue (min-heap) for nodes to be evaluated, ordered by fscore

    # Add start node to the heap
    heapq.heappush(oheap, (fscore[start_node], start_node))

    while oheap: # While there are nodes to evaluate
        # Get the node in the open set with the lowest fscore
        current = heapq.heappop(oheap)[1]

        # If we reached the end node, reconstruct and return the path
        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1] # Return reversed path (end -> start becomes start -> end)

        # Mark current node as evaluated
        close_set.add(current)

        # Explore neighbors
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            # Check if neighbor is within grid bounds
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue

            # Get the cost of moving to the neighbor cell
            neighbor_cost = grid[neighbor[1]][neighbor[0]]

            # Ignore impassable neighbors or neighbors already evaluated
            if neighbor_cost == OBSTACLE_COST or neighbor in close_set:
                continue

            # Calculate the cost to reach the neighbor through the current node
            tentative_g_score = gscore[current] + neighbor_cost

            # If this path to neighbor is better than any previous one, record it
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current # Update path parent
                gscore[neighbor] = tentative_g_score # Update cost from start
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end_node) # Update total estimated cost
                # Add neighbor to the heap to be evaluated.
                # If neighbor was already in heap, this adds a duplicate with lower cost.
                # heapq will process the lower cost one first.
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    # If the loop finishes without reaching the end_node, no path exists
    return None


# --- Entity Classes ---

class Obstacle:
    """Represents an impassable obstacle/wall on the grid."""
    def __init__(self, grid_x, grid_y):
        """
        Initializes an obstacle at a specific grid cell.
        Args:
            grid_x (int): Grid x-coordinate.
            grid_y (int): Grid y-coordinate.
        """
        self.grid_pos = (grid_x, grid_y)
        # Calculate the screen rectangle for drawing
        self.rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        self.color = OBSTACLE_COLOR

    def draw(self, screen):
        """Draws the obstacle on the screen if it's within the simulation area."""
        # Avoid drawing parts that spill into the UI panel
        if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Base:
    """Represents the central base where agents return resources, eat, and rest."""
    def __init__(self, x, y):
        """
        Initializes the base at given world coordinates.
        Args:
            x (float): World x-coordinate for the center of the base.
            y (float): World y-coordinate for the center of the base.
        """
        self.pos = pygame.Vector2(x, y)
        # Define the rectangle for drawing and collision
        self.rect = pygame.Rect(x - BASE_SIZE // 2, y - BASE_SIZE // 2, BASE_SIZE, BASE_SIZE)
        # Determine the primary grid cell associated with the base (used for pathfinding target)
        self.grid_pos = world_to_grid(x, y)
        self.color = BASE_COLOR

    def draw(self, screen):
        """Draws the base on the screen if it's within the simulation area."""
        if self.rect.right <= SIM_WIDTH and self.rect.bottom <= SIM_HEIGHT:
            pygame.draw.rect(screen, self.color, self.rect)

class Resource:
    """Represents a resource node that agents can collect from."""
    _id_counter = 0 # Class variable for generating unique IDs

    def __init__(self, x, y):
        """
        Initializes a resource node at given world coordinates.
        Args:
            x (float): World x-coordinate for the center of the resource.
            y (float): World y-coordinate for the center of the resource.
        Raises:
            ValueError: If the resource is created outside the simulation bounds.
        """
        self.id = Resource._id_counter
        Resource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = world_to_grid(x, y) # Associated grid cell

        # Ensure resource is placed within the valid simulation grid
        if self.grid_pos is None:
            raise ValueError(f"Resource created outside sim bounds: ({x},{y})")

        # Initialize quantity and regeneration properties
        self.quantity = random.randint(RESOURCE_MAX_QUANTITY // 2, RESOURCE_MAX_QUANTITY)
        self.max_quantity = RESOURCE_MAX_QUANTITY
        self.regen_timer = 0 # Timer for resource regeneration
        # Define the rectangle for drawing and quadtree insertion
        self.rect = pygame.Rect(self.pos.x - RESOURCE_SIZE // 2, self.pos.y - RESOURCE_SIZE // 2, RESOURCE_SIZE, RESOURCE_SIZE)

    def collect(self, amount=1):
        """
        Reduces the resource quantity when collected by an agent.
        Args:
            amount (int): The amount of resource to attempt collecting.
        Returns:
            int: The actual amount of resource collected (limited by quantity).
        """
        collected = min(amount, self.quantity)
        self.quantity -= collected
        self.quantity = max(0, self.quantity) # Ensure quantity doesn't go below zero
        return collected

    def update(self, dt):
        """
        Updates the resource node, handling regeneration over time.
        Args:
            dt (float): The time elapsed since the last update (delta time).
        """
        # Regenerate resource if below maximum quantity
        if self.quantity < self.max_quantity:
            self.regen_timer += dt
            # Calculate regeneration interval (avoid division by zero)
            regen_interval = 1.0 / RESOURCE_REGEN_RATE if RESOURCE_REGEN_RATE > 0 else float('inf')
            # If enough time has passed, regenerate one unit
            if self.regen_timer >= regen_interval:
                self.quantity = min(self.max_quantity, self.quantity + 1)
                self.regen_timer -= regen_interval # Reset timer partially or fully

    def draw(self, screen, is_targeted):
        """
        Draws the resource node on the screen. Color indicates quantity or targeted status.
        Args:
            screen (pygame.Surface): The display surface.
            is_targeted (bool): True if an agent is currently targeting this resource.
        """
        # Don't draw if outside the simulation area
        if self.rect.right > SIM_WIDTH or self.rect.bottom > SIM_HEIGHT: return

        color = DARK_GREY # Default color for empty resource
        if is_targeted:
            color = YELLOW # Highlight targeted resources
        elif self.quantity > 0:
            # Color gradient from red (low) to green (high) based on quantity
            ratio = self.quantity / self.max_quantity
            red_val = int(255 * (1 - ratio))
            green_val = int(255 * ratio)
            color = (red_val, green_val, 0)

        pygame.draw.rect(screen, color, self.rect)


# --- Agent Class ---
class Agent:
    """Represents an autonomous agent in the simulation."""
    _id_counter = 0 # Class variable for generating unique IDs

    def __init__(self, x, y, environment):
        """
        Initializes an agent at given world coordinates.
        Args:
            x (float): Initial world x-coordinate.
            y (float): Initial world y-coordinate.
            environment (Environment): Reference to the main environment object.
        """
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.environment = environment # Reference to access grid, resources, base etc.

        # Movement and Physics
        self.pos = pygame.Vector2(x, y)
        # Start with a small random velocity
        self.velocity = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
        self.max_speed = AGENT_SPEED
        self.max_force = AGENT_MAX_FORCE # Max steering force magnitude

        # State Machine and Task Management
        self.color = BLUE # Default draw color
        self.state = STATE_IDLE # Initial state
        self.action_type = None # Specific action if state is WORKING
        self.action_timer = 0 # Timer for WORKING state actions
        self.target_resource = None # Reference to the targeted Resource object
        self.target_pos_world = None # Current world position target (usually a path node)
        self.destination_grid_pos = None # Final grid destination of the current path

        # Pathfinding
        self.current_path = [] # List of grid coordinates [(x,y), ...] for the current path
        self.path_index = 0 # Index of the next node in current_path to move towards

        # Needs
        self.energy = MAX_ENERGY * random.uniform(0.8, 1.0) # Start with high energy
        self.hunger = MAX_HUNGER * random.uniform(0.0, 0.2) # Start with low hunger

        # Inventory
        self.carrying_resource = 0 # Amount of resource currently carried

        # Ensure initial position is within simulation bounds
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))

        # Rectangle for drawing and quadtree insertion (updated frequently)
        self.rect = pygame.Rect(self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE, AGENT_SIZE * 2, AGENT_SIZE * 2)
        self.update_rect() # Set initial rect center


    def update_rect(self):
        """Updates the agent's rectangle based on its current position."""
        self.rect.center = (int(self.pos.x), int(self.pos.y))

    def apply_force(self, force):
        """Applies a force vector to the agent's velocity. (Note: Directly added in update)."""
        # This method is conceptually useful but force is applied directly in update loop for simplicity
        self.velocity += force

    def seek(self, target_pos):
        """Calculates a steering force to move towards a target position."""
        if target_pos is None: return pygame.Vector2(0, 0) # No target, no force

        # Calculate desired velocity vector (points from agent to target)
        desired = target_pos - self.pos
        dist_sq = desired.length_squared()

        # If very close to the target, stop seeking (prevents jittering)
        if dist_sq < (GRID_CELL_SIZE * 0.4)**2: # Arrival threshold based on grid size
             return pygame.Vector2(0, 0)

        # Scale desired velocity to maximum speed
        if dist_sq > 0 : # Avoid normalizing zero vector
            desired.normalize_ip()
            desired *= self.max_speed
        else:
            return pygame.Vector2(0, 0) # Already at target

        # Calculate steering force = desired velocity - current velocity
        steer = desired - self.velocity

        # Limit the steering force to the maximum allowed force
        if steer.length_squared() > self.max_force * self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def separation(self, neighbors):
        """Calculates a steering force to avoid crowding nearby agents."""
        steer = pygame.Vector2(0, 0)
        count = 0
        # Define the desired minimum separation distance (slightly more than agent diameter)
        desired_separation = (AGENT_SIZE * 2) * 1.5

        for other in neighbors:
            if other is self: continue # Don't compare with self
            dist_sq = self.pos.distance_squared_to(other.pos)

            # If the other agent is within the desired separation distance
            if 0 < dist_sq < desired_separation * desired_separation:
                # Calculate repulsion vector (points away from the other agent)
                diff = self.pos - other.pos
                # Weight the repulsion force by inverse square distance (stronger when closer)
                # Add small epsilon to avoid division by zero if agents are exactly at the same spot
                diff *= (1.0 / (dist_sq + 0.001))
                steer += diff
                count += 1

        # Average the steering vector if multiple neighbors were too close
        if count > 0:
            steer /= count
            # Ensure the average steering vector is not zero before normalizing
            if steer.length_squared() > 0:
                # Scale the average direction to represent a desired velocity
                steer.normalize_ip()
                steer *= self.max_speed
                # Calculate the final steering force needed
                steer -= self.velocity
                # Limit the separation force
                if steer.length_squared() > self.max_force * self.max_force:
                    steer.scale_to_length(self.max_force)
        return steer

    def _find_path(self, target_grid_pos):
        """
        Attempts to find a path to a target grid cell using A*.
        Updates the agent's path variables if successful.

        Args:
            target_grid_pos (tuple[int, int]): The destination grid cell (x, y).

        Returns:
            bool: True if a path was found or already at destination, False otherwise.
        """
        # Get current grid position
        start_grid_pos = world_to_grid(self.pos.x, self.pos.y)

        # Reset current path state
        self.current_path = []
        self.path_index = 0
        self.target_pos_world = None
        self.destination_grid_pos = target_grid_pos # Store the final destination

        # Check validity of start/target positions
        if start_grid_pos is None or target_grid_pos is None:
             # print(f"Agent {self.id}: Pathing failed - start or target outside grid.") # Debug
             return False
        if start_grid_pos == target_grid_pos:
            # print(f"Agent {self.id}: Pathing skipped - already at destination.") # Debug
            return True # Already at the destination

        # Perform A* pathfinding using the environment's grid
        path = astar_pathfinding(self.environment.grid, start_grid_pos, target_grid_pos)

        # If A* found a path
        if path:
            self.current_path = path
            self.path_index = 0
            # Set the first node in the path as the immediate world target
            next_grid_node = self.current_path[self.path_index]
            self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            # print(f"Agent {self.id}: Path found to {target_grid_pos}. Length: {len(path)}") # Debug
            return True
        else:
            # print(f"Agent {self.id}: Pathing failed - A* could not find path to {target_grid_pos}.") # Debug
            self.destination_grid_pos = None # Clear destination if path failed
            return False

    def update(self, dt, nearby_agents):
        """
        Main update logic for the agent, called each frame.
        Handles needs, state machine, steering, movement, and boundary checks.

        Args:
            dt (float): Delta time for frame-rate independent movement.
            nearby_agents (list[Agent]): List of agents close to this one (from quadtree).
        """
        # --- Needs Update ---
        # Apply decay/increase based on time elapsed
        self.energy -= ENERGY_DECAY_RATE * dt
        self.hunger += HUNGER_INCREASE_RATE * dt
        # Clamp needs to their max/min values
        self.energy = max(0, self.energy)
        self.hunger = min(MAX_HUNGER, self.hunger)

        # Check for agent 'death' (out of energy)
        if self.energy <= 0:
            # print(f"Agent {self.id}: Died of exhaustion!") # Can be noisy
            self.environment.remove_agent(self) # Request removal from environment
            return # Stop further updates for this agent

        # --- Steering Forces Calculation ---
        seek_force = pygame.Vector2(0, 0) # Force towards current target
        # Calculate separation force based on nearby agents (increase weight for stronger effect)
        separation_force = self.separation(nearby_agents) * 1.5 # Weight separation higher

        # --- High Priority Need Checks & State Overrides ---
        needs_override = False # Flag to indicate if a need forced a state change
        # Check if currently moving to base specifically for needs
        is_moving_for_needs = self.state == STATE_MOVING_TO_BASE and (self.action_type == ACTION_EATING or self.action_type == ACTION_RESTING)

        # Check Hunger: If critically hungry and not already dealing with needs
        if self.hunger >= HIGH_HUNGER_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             # Check if base exists and agent is not already moving to base
             if self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 # Try to find a path to the base
                 if self._find_path(self.environment.base.grid_pos):
                     # print(f"Agent {self.id}: Critically hungry! Seeking base.") # Debug
                     self.state = STATE_MOVING_TO_BASE # Set state
                     self.action_type = ACTION_EATING   # Set intention
                     self._release_resource_target()    # Drop any current resource target
                     needs_override = True              # Mark that needs took priority
                 # else: print(f"Agent {self.id}: Hungry, cannot path to base!") # Debug

        # Check Energy: If critically tired, not dealing with hunger, and not already dealing with needs
        elif self.energy <= LOW_ENERGY_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             # Check if hunger didn't override, base exists, and not already moving to base
             if not needs_override and self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 # Try to find a path to the base
                 if self._find_path(self.environment.base.grid_pos):
                     # print(f"Agent {self.id}: Critically tired! Seeking base.") # Debug
                     self.state = STATE_MOVING_TO_BASE # Set state
                     self.action_type = ACTION_RESTING  # Set intention
                     self._release_resource_target()    # Drop any current resource target
                     needs_override = True              # Mark that needs took priority
                 # else: print(f"Agent {self.id}: Tired, cannot path to base!") # Debug

        # --- State Machine & Path Following Logic ---
        path_completed_this_frame = False # Flag if path ended in this update step

        # --- Path Following ---
        # If the agent has a path to follow
        if self.current_path:
            # If there's a valid world target position for the current path node
            if self.target_pos_world:
                 # Calculate distance to the current path node target
                 dist_to_node_sq = self.pos.distance_squared_to(self.target_pos_world)
                 # Define arrival threshold (squared for efficiency) - e.g., half a grid cell
                 arrival_threshold_sq = (GRID_CELL_SIZE * 0.5)**2

                 # If agent has reached the current path node
                 if dist_to_node_sq < arrival_threshold_sq:
                     self.path_index += 1 # Move to the next node index
                     # If there are more nodes left in the path
                     if self.path_index < len(self.current_path):
                         next_node_grid = self.current_path[self.path_index]
                         # Update world target to the center of the next grid node
                         self.target_pos_world = pygame.Vector2(grid_to_world_center(next_node_grid[0], next_node_grid[1]))
                     else: # Reached the end of the path
                         self.current_path = [] # Clear the path
                         self.target_pos_world = None # Clear world target
                         path_completed_this_frame = True # Set flag
                         # Optionally snap agent position to the center of the final grid destination
                         if self.destination_grid_pos:
                             self.pos = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))
                         self.velocity *= 0.1 # Dampen velocity significantly upon arrival
            else: # Edge case: Path exists but no target_pos_world? Clear path.
                self.current_path = []
                self.path_index = 0

        # --- Determine Seek Target ---
        # Default seek target is the current world path node
        seek_target = self.target_pos_world
        # If path just completed, seek the final destination grid cell center directly
        if path_completed_this_frame and self.destination_grid_pos:
             seek_target = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))

        # Calculate seek force towards the determined target
        if seek_target:
             seek_force = self.seek(seek_target)

        # --- State Logic (only if needs didn't override) ---
        if not needs_override:
            # --- IDLE State: Decide next action ---
            if self.state == STATE_IDLE:
                # Priority 1: Return resources if carrying any
                if self.carrying_resource > 0:
                    if self.environment.base and self._find_path(self.environment.base.grid_pos):
                         self.state = STATE_MOVING_TO_BASE
                         self.action_type = ACTION_RETURNING
                # Priority 2: Consider building (less frequent, requires resources and energy)
                elif self.carrying_resource >= BUILD_COST and self.energy > LOW_ENERGY_THRESHOLD + 20 and random.random() < 0.01: # Low probability
                    build_spot = self._find_build_spot() # Find adjacent buildable spot
                    if build_spot and self._find_path(build_spot): # Path to the build spot
                         self.state = STATE_MOVING_TO_BUILD
                # Priority 3: Find a resource to collect
                else:
                    resource_target = self._find_best_available_resource()
                    if resource_target and self._find_path(resource_target.grid_pos):
                        self.state = STATE_MOVING_TO_RESOURCE
                        self.target_resource = resource_target
                        # Mark the resource as targeted in the environment
                        self.environment.mark_resource_targeted(resource_target, self)

            # --- Moving States: Transition upon path completion ---
            elif self.state == STATE_MOVING_TO_RESOURCE and path_completed_this_frame:
                # Check if the targeted resource is still valid and targeted by *this* agent
                if self.target_resource and self.target_resource.quantity > 0 and self.environment.is_resource_targeted_by(self.target_resource, self):
                    self.state = STATE_WORKING        # Start working
                    self.action_type = ACTION_COLLECTING # Set action type
                    # Set timer for collection (can add randomness or base on resource type/quantity)
                    self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                else: # Resource depleted or taken by another agent before arrival
                    self._release_resource_target() # Release the target
                    self.state = STATE_IDLE         # Go back to idle to decide again

            elif self.state == STATE_MOVING_TO_BUILD and path_completed_this_frame:
                current_grid_pos = world_to_grid(self.pos.x, self.pos.y)
                build_target_grid = self.destination_grid_pos
                # Check if arrived at the intended build spot and it's still valid/buildable
                if build_target_grid and current_grid_pos == build_target_grid and \
                   self.environment.is_buildable(build_target_grid[0], build_target_grid[1]):
                    self.state = STATE_WORKING
                    self.action_type = ACTION_BUILDING
                    self.action_timer = BUILD_TIME
                    # Keep destination_grid_pos to know where to build when timer finishes
                else: # Arrived but target changed, became invalid, or wasn't the intended spot?
                    self.destination_grid_pos = None # Clear the build target
                    self.state = STATE_IDLE # Go back to idle

            elif self.state == STATE_MOVING_TO_BASE and path_completed_this_frame:
                 # Check the reason for going to the base
                 if self.action_type == ACTION_RETURNING:
                     # Drop off resources at the base
                     self.environment.add_base_resources(self.carrying_resource)
                     self.carrying_resource = 0
                     self.action_type = None # Clear action type
                     self.state = STATE_IDLE # Re-evaluate needs/tasks
                 elif self.action_type == ACTION_EATING:
                     # Start the eating action timer
                     self.state = STATE_WORKING
                     self.action_timer = EAT_TIME
                     # action_type remains ACTION_EATING
                 elif self.action_type == ACTION_RESTING:
                     # Start the resting action timer
                     self.state = STATE_WORKING
                     self.action_timer = REST_TIME
                     # action_type remains ACTION_RESTING
                 else: # Arrived at base for an unknown or completed reason
                     self.action_type = None
                     self.state = STATE_IDLE

            # --- WORKING State: Perform timed action ---
            elif self.state == STATE_WORKING:
                self.action_timer -= dt # Count down the timer
                # If timer finished
                if self.action_timer <= 0:
                    action_done = False # Flag to check if action completed successfully
                    # --- Perform action based on action_type ---
                    if self.action_type == ACTION_COLLECTING:
                        # Check if target resource is still valid
                        if self.target_resource and self.target_resource.quantity > 0:
                            collected = self.target_resource.collect(1) # Collect one unit
                            self.carrying_resource += collected
                            # Decide whether to continue collecting or stop
                            # Stop if resource empty or agent reaches simple capacity limit
                            if self.target_resource.quantity == 0 or self.carrying_resource >= 5:
                                self._release_resource_target() # Release target
                                action_done = True
                            else: # Continue collecting - reset timer for next unit
                                self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                        else: # Resource depleted or lost target while collecting
                            self._release_resource_target()
                            action_done = True

                    elif self.action_type == ACTION_BUILDING:
                         build_pos = self.destination_grid_pos # Where agent intended to build
                         # Check if build position is still valid and have resources
                         if build_pos and self.carrying_resource >= BUILD_COST:
                              # Attempt to build the wall via the environment
                              build_success = self.environment.build_wall(build_pos[0], build_pos[1], self)
                              if build_success:
                                   self.carrying_resource -= BUILD_COST # Consume resources
                         # else: Build failed (spot became invalid) or not enough resources
                         self.destination_grid_pos = None # Clear build target regardless of success/failure
                         action_done = True

                    elif self.action_type == ACTION_EATING:
                         # Reduce hunger
                         self.hunger = max(0, self.hunger - EAT_AMOUNT)
                         action_done = True
                    elif self.action_type == ACTION_RESTING:
                         # Restore energy
                         self.energy = min(MAX_ENERGY, self.energy + REST_AMOUNT)
                         action_done = True
                    else: # Unknown or completed action type
                         action_done = True

                    # If the timed action is finished, reset state to IDLE
                    if action_done:
                         self.action_type = None
                         self.state = STATE_IDLE

        # --- Apply Forces & Update Movement ---
        # Combine steering forces
        total_force = seek_force + separation_force

        # Apply force to velocity (Euler integration: v += F * dt)
        # (Assuming mass = 1, so acceleration = force)
        self.velocity += total_force * dt

        # Limit overall velocity to max_speed
        vel_mag_sq = self.velocity.length_squared()
        if vel_mag_sq > self.max_speed * self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        # Apply damping based on state
        elif self.state == STATE_WORKING:
             self.velocity *= 0.05 # Slow down significantly while working
        elif self.state == STATE_IDLE and not self.target_pos_world and not self.current_path:
             # Slow down gradually if idle and not actively moving towards a path node
             self.velocity *= 0.85

        # Update position based on velocity (Euler integration: p += v * dt)
        self.pos += self.velocity * dt

        # --- Boundary Constraints (Keep agent within SIMULATION AREA) ---
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))

        # Update the agent's rectangle for drawing and quadtree
        self.update_rect()


    def _release_resource_target(self):
        """Informs the environment that this agent is no longer targeting its current resource."""
        if self.target_resource:
            self.environment.mark_resource_available(self.target_resource)
            self.target_resource = None # Clear agent's own reference

    def _find_best_available_resource(self, max_search_radius=300):
        """
        Finds the 'best' available and untargeted resource within a search radius.
        'Best' is determined by a score considering quantity and distance.

        Args:
            max_search_radius (float): The maximum distance to search for resources.

        Returns:
            Resource or None: The best found resource object, or None if none found/available.
        """
        best_score = -float('inf') # Initialize score to find maximum
        best_resource = None

        # Use Quadtree to efficiently find resources within the search radius
        potential_resources = self.environment.quadtree.query_radius(self.pos, max_search_radius)

        # Filter the results: must be a Resource, have quantity > 0, and not be targeted by another agent
        available_resources = [
            res for res in potential_resources
            if isinstance(res, Resource) and res.quantity > 0 and res.id not in self.environment.targeted_resources
        ]

        if not available_resources: return None # No suitable resources found nearby

        # Define weights for scoring resources (can be tuned)
        weight_quantity = 1.5 # Higher quantity is better
        weight_distance = -0.05 # Closer distance is better (negative weight)

        # Evaluate each available resource
        for resource in available_resources:
            dist_sq = self.pos.distance_squared_to(resource.pos)
            # Use sqrt only if needed, or work with squared values if comparison allows
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1 # Avoid division by zero/sqrt(0)

            # Calculate score based on weighted quantity and distance
            score = (weight_quantity * resource.quantity) + (weight_distance * dist)

            # Update best resource if current one has a higher score
            if score > best_score:
                 # Optimization Note: Could add an A* check here to ensure pathability,
                 # but it can be computationally expensive to do for every potential resource.
                 # Path check is currently done *after* selecting the target in the main state machine.
                 # if astar_pathfinding(self.environment.grid, world_to_grid(*self.pos), resource.grid_pos):
                 best_score = score
                 best_resource = resource
                 # else: Resource inaccessible, ignore.

        return best_resource


    def _find_build_spot(self):
        """Finds a valid, adjacent grid cell where the agent can build a wall."""
        my_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        if my_grid_pos is None: return None # Agent is somehow outside the grid

        # Define adjacent cell offsets
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Down, Up, Right, Left
        random.shuffle(neighbors) # Check neighbors in random order

        for dx, dy in neighbors:
            check_x, check_y = my_grid_pos[0] + dx, my_grid_pos[1] + dy
            # Use environment's method to check if the cell is buildable
            if self.environment.is_buildable(check_x, check_y):
                 return (check_x, check_y) # Return the first valid spot found

        return None # No adjacent buildable spot found

    def draw(self, screen):
        """Draws the agent on the screen, including state indicators and needs bars."""
        # Don't draw if the agent is outside the visible simulation area
        if self.pos.x >= SIM_WIDTH or self.pos.y >= SIM_HEIGHT or self.pos.x < 0 or self.pos.y < 0:
            return

        # --- Draw Base Circle ---
        center_tuple = (int(self.pos.x), int(self.pos.y))
        pygame.draw.circle(screen, self.color, center_tuple, AGENT_SIZE)

        # --- Draw State Indicator (Inner Circle) ---
        state_color = WHITE # Default for IDLE
        current_state = self.state
        current_action = self.action_type

        # Determine color based on state and action type
        if current_state == STATE_MOVING_TO_RESOURCE: state_color = CYAN
        elif current_state == STATE_MOVING_TO_BUILD: state_color = ORANGE
        elif current_state == STATE_MOVING_TO_BASE:
            if current_action == ACTION_RETURNING: state_color = YELLOW
            elif current_action == ACTION_EATING: state_color = RED
            elif current_action == ACTION_RESTING: state_color = GREEN
            else: state_color = PURPLE # Generic moving to base
        elif current_state == STATE_WORKING:
            if current_action == ACTION_COLLECTING: state_color = ORANGE
            elif current_action == ACTION_BUILDING: state_color = GREY
            elif current_action == ACTION_EATING: state_color = RED
            elif current_action == ACTION_RESTING: state_color = GREEN
            else: state_color = DARK_GREY # Generic working
        # Draw the inner circle indicating state
        pygame.draw.circle(screen, state_color, center_tuple, max(1, AGENT_SIZE // 2))

        # --- Draw Needs Bars below agent ---
        bar_width = AGENT_SIZE * 2
        bar_height = 3
        bar_padding = 1
        bar_x = self.pos.x - bar_width / 2
        bar_y_energy = self.pos.y + AGENT_SIZE + 2 # Position below agent circle
        bar_y_hunger = bar_y_energy + bar_height + bar_padding # Position below energy bar

        # Energy Bar (Green)
        energy_ratio = self.energy / MAX_ENERGY
        pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_energy, bar_width, bar_height)) # Background
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y_energy, bar_width * energy_ratio, bar_height)) # Foreground

        # Hunger Bar (Red) - Higher value is worse
        hunger_ratio = self.hunger / MAX_HUNGER
        pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_hunger, bar_width, bar_height)) # Background
        pygame.draw.rect(screen, RED, (bar_x, bar_y_hunger, bar_width * hunger_ratio, bar_height)) # Foreground

# --- UI Button Class ---
class Button:
    """Represents a clickable button in the UI panel."""
    def __init__(self, x, y, w, h, text, action, font):
        """
        Initializes a UI button.
        Args:
            x, y (int): Top-left coordinates of the button.
            w, h (int): Width and height of the button.
            text (str): Text displayed on the button.
            action (any): An identifier or tuple representing the action performed on click.
            font (pygame.font.Font): The font used for rendering the button text.
        """
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action # Action associated with the button (e.g., string 'pause', tuple ('set_brush', BRUSH_WALL))
        self.font = font
        self.is_hovered = False # True if mouse is over the button
        self.is_selected = False # True if button is a toggle and currently active (e.g., current paint brush)

    def draw(self, screen):
        """Draws the button on the screen."""
        color = BUTTON_COLOR # Default color
        # Change color based on state
        if self.is_selected:
            color = SELECTED_BUTTON_COLOR
        elif self.is_hovered:
            color = BUTTON_HOVER_COLOR

        # Draw button background and border
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 1) # White border

        # Render and center the text on the button
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        """Updates the hover state based on the mouse position."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_click(self):
        """Checks if the button was clicked (called when mouse click occurs)."""
        if self.is_hovered:
            # print(f"Button Clicked: {self.action}") # Debug
            return self.action # Return the associated action identifier
        return None

# --- Environment Class ---
class Environment:
    """
    Manages the simulation state, including agents, resources, obstacles,
    the grid, quadtree, base, and UI interactions.
    """
    def __init__(self, width, height, sim_width, sim_height, font_small, font_medium):
        """
        Initializes the environment.
        Args:
            width, height (int): Total screen dimensions.
            sim_width, sim_height (int): Dimensions of the simulation area.
            font_small, font_medium (pygame.font.Font): Fonts for UI rendering.
        """
        self.width = width
        self.height = height
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.font_small = font_small # Font references (need special handling for save/load)
        self.font_medium = font_medium

        # Entity Management
        self.agents = []
        self.resources = []
        self.obstacles = [] # Primarily used for drawing if needed; grid holds pathfinding data
        self.base = None
        self.total_base_resources = 0 # Resources deposited at the base
        self.targeted_resources = {} # {resource_id: agent_id} mapping

        # World Representation
        # Initialize grid with default terrain (e.g., Plains)
        self.grid = [[TERRAIN_PLAINS for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        # Initialize Quadtree covering the simulation area
        self.quadtree = Quadtree(0, pygame.Rect(0, 0, sim_width, sim_height))

        # UI State
        self.selected_agent = None # Currently selected agent for detailed view
        self.speed_multiplier = 1.0 # Simulation speed control
        self.paint_brush = BRUSH_SELECT # Currently active painting/selection tool
        self.buttons = [] # List to hold UI Button objects
        self._setup_ui_buttons() # Create the UI buttons


    def _setup_ui_buttons(self):
        """Creates and configures all the Button objects for the UI panel."""
        self.buttons = [] # Clear existing buttons if any
        button_font = self.font_small # Use the small font for buttons
        # Calculate button width for two columns, considering padding
        bw = (UI_PANEL_WIDTH - 3 * BUTTON_PADDING) // 2
        bh = BUTTON_HEIGHT # Standard button height
        bx = UI_PANEL_X + BUTTON_PADDING # Starting X for buttons within the panel
        by = 10 # Starting Y coordinate

        # --- Control Buttons (Pause, Play, Speed) ---
        self.buttons.append(Button(bx, by, bw, bh, "Pause", "toggle_pause", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Play", "play", button_font))
        by += bh + BUTTON_PADDING # Move Y down for next row
        self.buttons.append(Button(bx, by, bw, bh, "Slow (-)", "speed_down", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Fast (+)", "speed_up", button_font))
        by += bh + BUTTON_PADDING
        # Full width button for Reset Speed
        self.buttons.append(Button(bx, by, bw*2 + BUTTON_PADDING, bh, "Reset Speed", "speed_reset", button_font))
        by += bh + BUTTON_PADDING * 3 # Add extra spacing

        # --- Toggle Buttons (Debug Views) ---
        self.buttons.append(Button(bx, by, bw, bh, "Grid", "toggle_grid", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Quadtree", "toggle_quadtree", button_font))
        by += bh + BUTTON_PADDING * 3 # Add extra spacing

        # --- Paint Brush Buttons ---
        brush_bw = UI_PANEL_WIDTH - 2 * BUTTON_PADDING # Use full panel width for brush buttons
        # Select Tool (technically a brush)
        self.buttons.append(Button(bx, by, brush_bw, bh, "Select Tool", ("set_brush", BRUSH_SELECT), button_font))
        by += bh + BUTTON_PADDING
        # Terrain/Wall Painting Brushes
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Wall", ("set_brush", BRUSH_WALL), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Clear", ("set_brush", BRUSH_CLEAR), button_font)) # Clears to Plains
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Plains", ("set_brush", BRUSH_PLAINS), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Grass", ("set_brush", BRUSH_GRASS), button_font))
        by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, brush_bw, bh, "Paint Mud", ("set_brush", BRUSH_MUD), button_font))
        by += bh + BUTTON_PADDING * 3 # Add extra spacing

        # --- Save/Load Buttons ---
        self.buttons.append(Button(bx, by, bw, bh, "Save (F5)", "save", button_font))
        self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Load (F9)", "load", button_font))

        # Ensure the initially selected brush button is highlighted
        self._update_button_selected_state()

    def _update_button_selected_state(self):
        """Updates the 'is_selected' state of toggle buttons (like paint brushes)."""
        for button in self.buttons:
            # Check if the button action is for setting a brush
            if isinstance(button.action, tuple) and button.action[0] == "set_brush":
                # Set selected state if the button's brush matches the environment's current brush
                button.is_selected = (button.action[1] == self.paint_brush)
            else:
                button.is_selected = False # Ensure other buttons are not marked selected

    def handle_ui_click(self, mouse_pos, game_state):
        """
        Processes a mouse click within the UI panel, checks for button presses,
        and performs the corresponding action.

        Args:
            mouse_pos (tuple[int, int]): The mouse click coordinates.
            game_state (dict): The main game state dictionary to potentially modify.

        Returns:
            str or bool or None: Returns "save" or "load" if those buttons were clicked,
                                 True if any other UI action was handled,
                                 False or None otherwise.
        """
        clicked_action = None
        # Check collision with each button
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                clicked_action = button.handle_click() # Get action from the clicked button
                break # Handle only the first button clicked

        # Perform action based on the clicked button's identifier
        if clicked_action:
            if clicked_action == "toggle_pause": game_state['paused'] = not game_state['paused']
            elif clicked_action == "play": game_state['paused'] = False
            elif clicked_action == "speed_down": self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5) # Limit minimum speed
            elif clicked_action == "speed_up": self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5) # Limit maximum speed
            elif clicked_action == "speed_reset": self.speed_multiplier = 1.0
            elif clicked_action == "toggle_grid": game_state['draw_grid'] = not game_state['draw_grid']
            elif clicked_action == "toggle_quadtree": game_state['draw_quadtree'] = not game_state['draw_quadtree']
            elif clicked_action == "save": return "save" # Signal main loop to save
            elif clicked_action == "load": return "load" # Signal main loop to load
            # Handle brush selection
            elif isinstance(clicked_action, tuple) and clicked_action[0] == "set_brush":
                self.paint_brush = clicked_action[1] # Update the environment's brush
                self._update_button_selected_state() # Update button visuals
            return True # Indicate that the UI handled the click
        return False # UI did not handle the click

    def paint_terrain(self, mouse_pos, brush_size=1):
        """
        Modifies the grid terrain based on the current paint brush and mouse position.
        Applies paint in a square area defined by brush_size.

        Args:
            mouse_pos (tuple[int, int]): The mouse coordinates within the simulation area.
            brush_size (int): The width/height of the square brush (in grid cells).
        """
        # Convert mouse position to grid coordinates
        grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
        if grid_pos is None: return # Click was outside the simulation area

        gx, gy = grid_pos
        terrain_cost_to_paint = None
        is_clearing = False

        # Determine the terrain cost based on the selected brush
        if self.paint_brush == BRUSH_WALL:
            terrain_cost_to_paint = TERRAIN_WALL # Impassable cost
        elif self.paint_brush == BRUSH_CLEAR:
             terrain_cost_to_paint = TERRAIN_PLAINS # Clearing sets back to default walkable terrain
             is_clearing = True
        elif self.paint_brush == BRUSH_PLAINS:
             terrain_cost_to_paint = TERRAIN_PLAINS
        elif self.paint_brush == BRUSH_GRASS:
              terrain_cost_to_paint = TERRAIN_GRASS
        elif self.paint_brush == BRUSH_MUD:
               terrain_cost_to_paint = TERRAIN_MUD

        # If a valid brush is selected (not SELECT tool)
        if terrain_cost_to_paint is not None:
             # Apply paint in a square area centered around the click/drag point
             offset = brush_size // 2
             for dx in range(-offset, offset + 1):
                 for dy in range(-offset, offset + 1):
                      paint_x, paint_y = gx + dx, gy + dy
                      # Ensure the paint location is within grid bounds
                      if 0 <= paint_x < GRID_WIDTH and 0 <= paint_y < GRID_HEIGHT:
                           # --- Prevent painting over crucial entities ---
                           # Check if painting over the base's primary grid cell
                           is_base = self.base and self.base.grid_pos == (paint_x, paint_y)
                           # Check if painting over any resource's grid cell
                           is_res = any(r.grid_pos == (paint_x, paint_y) for r in self.resources)

                           # Allow painting only if not base/resource OR if clearing
                           # (Clearing should ideally remove obstacles, but not base/resources)
                           if (not is_base and not is_res) or is_clearing:
                               # Apply the terrain cost to the grid
                               self.grid[paint_y][paint_x] = terrain_cost_to_paint

                               # Optional: Update self.obstacles list if managing it separately
                               # If painting a wall, could add Obstacle object.
                               # If clearing, could find and remove Obstacle object at this pos.
                               # For simplicity, relying solely on the grid for pathfinding cost is sufficient.

    def is_buildable(self, grid_x, grid_y):
         """
         Checks if a wall can be built at the specified grid cell.
         Considers terrain type and existing entities.

         Args:
             grid_x (int): Grid x-coordinate.
             grid_y (int): Grid y-coordinate.

         Returns:
             bool: True if building is allowed, False otherwise.
         """
         # Check if coordinates are within grid bounds
         if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             # Get the current terrain cost at the location
             terrain_cost = self.grid[grid_y][grid_x]

             # --- Rules for buildability ---
             # 1. Cannot build on impassable terrain (existing walls, obstacles, maybe mud)
             if terrain_cost != OBSTACLE_COST and terrain_cost != TERRAIN_WALL and terrain_cost != TERRAIN_MUD:
                  # 2. Cannot build on the base's primary grid cell
                  if not (self.base and self.base.grid_pos == (grid_x, grid_y)):
                      # 3. Cannot build on a resource's grid cell
                      is_res_spot = any(r.grid_pos == (grid_x, grid_y) for r in self.resources)
                      if not is_res_spot:
                          # If all checks pass, it's buildable
                          return True
         # If any check failed or out of bounds
         return False

    # --- Entity Management Methods ---
    def add_agent(self, agent):
        """Adds an agent to the simulation."""
        self.agents.append(agent)
        # Note: Quadtree insertion happens during the rebuild phase

    def remove_agent(self, agent_to_remove):
        """Removes an agent from the simulation."""
        # Ensure agent releases any resource it was targeting
        agent_to_remove._release_resource_target()
        # Remove agent from the main list
        self.agents = [a for a in self.agents if a.id != agent_to_remove.id]
        # If the removed agent was selected, deselect it
        if self.selected_agent and self.selected_agent.id == agent_to_remove.id:
             self.selected_agent = None

    def add_resource(self, resource):
        """Adds a resource node, checking validity."""
        if resource.grid_pos is None: return # Should have been caught by Resource.__init__
        gx, gy = resource.grid_pos
        # Only add resource if the grid cell is not an obstacle/wall
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.resources.append(resource)
        # else: Cannot place resource on an obstacle

    def add_obstacle(self, obstacle):
        """Adds an obstacle by updating the grid cost."""
        gx, gy = obstacle.grid_pos
        # Check bounds
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
             # Don't place obstacle if cell is already an obstacle
             if self.grid[gy][gx] != OBSTACLE_COST:
                  # Prevent placing obstacle on base or existing resources
                  if self.base and self.base.grid_pos == (gx, gy): return
                  if any(r.grid_pos == (gx, gy) for r in self.resources): return

                  # Update the grid to mark this cell as impassable
                  self.grid[gy][gx] = OBSTACLE_COST
                  # Optionally add to self.obstacles list if needed for other reasons
                  # self.obstacles.append(obstacle)

    def set_base(self, base):
        """Sets the simulation's base, checking validity."""
        if base.grid_pos is None: return # Base position invalid
        gx, gy = base.grid_pos
        # Ensure base is not placed on an existing obstacle
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.base = base
        else:
            print(f"Warning: Cannot set base at occupied grid cell {base.grid_pos}")

    def add_base_resources(self, amount):
        """Increases the total resources stored at the base."""
        self.total_base_resources += amount

    def build_wall(self, grid_x, grid_y, builder_agent):
        """
        Called by an agent to build a wall at a specific grid location.
        Checks buildability before modifying the grid.

        Args:
            grid_x, grid_y (int): Grid coordinates for the wall.
            builder_agent (Agent): The agent performing the build action.

        Returns:
            bool: True if the wall was successfully built, False otherwise.
        """
        # Check if the location is valid for building
        if self.is_buildable(grid_x, grid_y):
             # Update the grid cost to make it an impassable wall
             self.grid[grid_y][grid_x] = TERRAIN_WALL
             # print(f"Agent {builder_agent.id} built wall at ({grid_x}, {grid_y})") # Debug
             return True
        return False

    def create_random_entity(self, entity_type):
        """
        Attempts to create and place an agent or resource at a random valid location.

        Args:
            entity_type (str): 'agent' or 'resource'.
        """
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            attempts += 1
            # Choose random grid coordinates within the sim area (avoiding edges initially)
            gx = random.randint(1, GRID_WIDTH - 2)
            gy = random.randint(1, GRID_HEIGHT - 2)

            # Check if the chosen grid cell is 'buildable' (i.e., generally suitable for placement)
            if self.is_buildable(gx, gy):
                 # Convert grid position back to world coordinates (cell center)
                 wx, wy = grid_to_world_center(gx, gy)
                 try:
                     # Create and add the entity
                     if entity_type == 'agent':
                         self.add_agent(Agent(wx, wy, self))
                     elif entity_type == 'resource':
                         self.add_resource(Resource(wx, wy))
                     return # Success, exit loop
                 except ValueError as e: # Catch errors like resource outside bounds (shouldn't happen here)
                      print(f"Error creating entity during random placement: {e}")
                      # Continue trying...
            # If spot not buildable, loop continues to try another random spot

        # If max attempts reached without success
        print(f"Warning: Could not find a suitable spot for random '{entity_type}' after {max_attempts} attempts.")

    def create_obstacle_line(self, x1, y1, x2, y2):
        """Creates a line of obstacles on the grid between two world points using Bresenham's algorithm."""
        # Clamp world coordinates to ensure they are within the simulation area before converting to grid
        x1 = max(0, min(SIM_WIDTH - 1, x1)); y1 = max(0, min(SIM_HEIGHT - 1, y1))
        x2 = max(0, min(SIM_WIDTH - 1, x2)); y2 = max(0, min(SIM_HEIGHT - 1, y2))

        # Convert potentially clamped world coordinates to grid coordinates
        start_node = world_to_grid(x1, y1)
        end_node = world_to_grid(x2, y2)

        # Exit if start or end is outside the grid after conversion
        if start_node is None or end_node is None: return

        # Bresenham's line algorithm adapted for grid cells
        gx1, gy1 = start_node
        gx2, gy2 = end_node
        dx = abs(gx2 - gx1)
        dy = -abs(gy2 - gy1) # Use negative dy for algorithm structure
        sx = 1 if gx1 < gx2 else -1 # Step direction for x
        sy = 1 if gy1 < gy2 else -1 # Step direction for y
        err = dx + dy # Initial error value

        while True:
             # Add an obstacle at the current grid cell (add_obstacle handles validity checks)
             self.add_obstacle(Obstacle(gx1, gy1))

             # Check if we reached the end point
             if gx1 == gx2 and gy1 == gy2: break

             # Calculate error for next step
             e2 = 2 * err
             # Step along x-axis if needed
             if e2 >= dy:
                 if gx1 == gx2: break # Avoid infinite loop if vertical line ends here
                 err += dy
                 gx1 += sx
             # Step along y-axis if needed
             if e2 <= dx:
                 if gy1 == gy2: break # Avoid infinite loop if horizontal line ends here
                 err += dx
                 gy1 += sy

    # --- Resource Targeting ---
    def mark_resource_targeted(self, resource, agent):
        """Marks a resource as being targeted by a specific agent."""
        if resource and resource.id not in self.targeted_resources:
            self.targeted_resources[resource.id] = agent.id

    def mark_resource_available(self, resource):
        """Marks a resource as no longer targeted."""
        if resource and resource.id in self.targeted_resources:
            del self.targeted_resources[resource.id]

    def is_resource_targeted_by(self, resource, agent):
        """Checks if a specific agent is the one currently targeting a resource."""
        return self.targeted_resources.get(resource.id) == agent.id

    # --- Agent Selection ---
    def select_agent_at(self, mouse_pos):
        """Selects the agent closest to the mouse click position within the simulation area."""
        # Ensure selection only happens within the simulation area boundaries
        if mouse_pos[0] >= self.sim_width or mouse_pos[1] >= self.sim_height or mouse_pos[0] < 0 or mouse_pos[1] < 0:
             self.selected_agent = None
             return

        self.selected_agent = None # Deselect previous agent first
        search_radius = AGENT_SIZE * 2.5 # Define a small search radius around the click

        # Use Quadtree to find potential agents near the click point
        nearby_entities = self.quadtree.query_radius(mouse_pos, search_radius)
        nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]

        # Find the closest agent among the nearby ones
        min_dist_sq = search_radius * search_radius # Initialize minimum distance squared
        closest_agent = None
        click_vec = pygame.Vector2(mouse_pos)
        for agent in nearby_agents:
             dist_sq = agent.pos.distance_squared_to(click_vec)
             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_agent = agent

        # Update the selected agent
        self.selected_agent = closest_agent
        # if self.selected_agent: print(f"Selected Agent {self.selected_agent.id}") # Debug

    # --- Main Update Methods ---
    def update_agents(self, dt):
         """Updates all agents in the simulation."""
         # Create a copy of the list to iterate over, allowing removal during iteration
         agents_to_update = self.agents[:]
         for agent in agents_to_update:
              # Check if agent still exists (might have been removed by another agent's update, e.g., death)
              if agent in self.agents:
                   # Define radius to search for neighbors for separation behavior
                   nearby_radius = (AGENT_SIZE * 2) * 3 # Search a few diameters away
                   # Query quadtree for nearby entities (potential neighbors)
                   nearby_entities = self.quadtree.query_radius(agent.pos, nearby_radius)
                   # Filter for actual agents
                   nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]
                   # Call the agent's update method, passing dt and neighbors
                   # Speed multiplier is applied here before passing dt to agent update
                   agent.update(dt * self.speed_multiplier, nearby_agents)

    def update_resources(self, dt):
        """Updates all resource nodes (handles regeneration)."""
        for resource in self.resources:
            # Speed multiplier is applied here before passing dt to resource update
            resource.update(dt * self.speed_multiplier)

    def rebuild_quadtree(self):
        """Clears and rebuilds the quadtree with current agent and resource positions."""
        self.quadtree.clear()
        # Insert agents (ensure they have updated rects and are within sim bounds)
        for agent in self.agents:
            # Double check bounds just before insertion
            if 0 <= agent.pos.x < self.sim_width and 0 <= agent.pos.y < self.sim_height:
                agent.update_rect() # Ensure rect is up-to-date
                self.quadtree.insert(agent)
        # Insert resources (ensure they have rects and are within sim bounds)
        for resource in self.resources:
             if 0 <= resource.pos.x < self.sim_width and 0 <= resource.pos.y < self.sim_height:
                # Ensure rect exists and is updated (should be set in __init__)
                if not hasattr(resource, 'rect'):
                    resource.rect = pygame.Rect(0,0,RESOURCE_SIZE, RESOURCE_SIZE)
                resource.rect.center = (int(resource.pos.x), int(resource.pos.y))
                self.quadtree.insert(resource)

    # --- Drawing Methods ---
    def draw_sim_area(self, screen, draw_grid_flag, draw_quadtree_flag):
        """
        Draws all elements within the simulation area (left side of the screen).
        Args:
            screen (pygame.Surface): The main display surface.
            draw_grid_flag (bool): Whether to draw the terrain grid.
            draw_quadtree_flag (bool): Whether to draw the quadtree boundaries (debug).
        """
        # 1. Draw Terrain Grid (if enabled) - This acts as the background
        if draw_grid_flag:
            self.draw_grid(screen)
        else:
            # Optional: Fill sim area with a default background color if grid isn't drawn
            sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
            screen.fill(BLACK, sim_rect) # Example: Black background

        # 2. Draw Base
        if self.base:
            self.base.draw(screen)

        # 3. Draw Obstacles (Optional - Walls are drawn by draw_grid if TERRAIN_WALL uses OBSTACLE_COLOR)
        # If you have separate Obstacle objects you want to draw distinctly:
        # for obs in self.obstacles: obs.draw(screen)

        # 4. Draw Resources
        for res in self.resources:
            res.draw(screen, res.id in self.targeted_resources) # Pass targeted status

        # 5. Draw Agents
        for agent in self.agents:
            agent.draw(screen) # Agent draw method handles bounds checks

        # 6. Draw Quadtree Boundaries (if enabled for debugging)
        if draw_quadtree_flag:
            self.quadtree.draw(screen)

        # 7. Draw Highlight for Selected Agent
        if self.selected_agent and self.selected_agent in self.agents:
             # Ensure highlight position is valid before drawing
             center_x = int(self.selected_agent.pos.x)
             center_y = int(self.selected_agent.pos.y)
             if 0 <= center_x < self.sim_width and 0 <= center_y < self.sim_height:
                 # Draw a white circle outline around the selected agent
                 pygame.draw.circle(screen, WHITE, (center_x, center_y), AGENT_SIZE + 3, 1)

    def draw_grid(self, screen):
        """Draws the terrain grid cells with their corresponding colors."""
        sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
        # Iterate through each grid cell
        for gy in range(GRID_HEIGHT):
            for gx in range(GRID_WIDTH):
                terrain_cost = self.grid[gy][gx] # Get the terrain cost/type
                # Get the color associated with this terrain type
                color = TERRAIN_COLORS.get(terrain_cost, GREY) # Default to GREY if cost unknown
                # Define the rectangle for this grid cell
                rect = pygame.Rect(gx * GRID_CELL_SIZE, gy * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)

                # Optimization: Only draw the rect if it's actually within the simulation area
                # (useful if grid theoretically extends beyond screen, though not the case here)
                if sim_rect.colliderect(rect): # Check for intersection
                    pygame.draw.rect(screen, color, rect)
                    # Optional: Draw grid lines for clarity
                    # pygame.draw.rect(screen, DARK_GREY, rect, 1)

    def draw_ui(self, screen, clock, game_state):
        """
        Draws the user interface panel on the right side of the screen.
        Args:
            screen (pygame.Surface): The main display surface.
            clock (pygame.time.Clock): Clock object for FPS display.
            game_state (dict): The main game state dictionary (for pause status etc.).
        """
        # --- Draw UI Panel Background ---
        ui_panel_rect = pygame.Rect(UI_PANEL_X, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, UI_BG_COLOR, ui_panel_rect)
        # Draw a vertical line separating sim area and UI panel
        pygame.draw.line(screen, WHITE, (UI_PANEL_X, 0), (UI_PANEL_X, SCREEN_HEIGHT), 1)

        # --- Draw Buttons ---
        mouse_pos = pygame.mouse.get_pos() # Get current mouse position
        for button in self.buttons:
            button.check_hover(mouse_pos) # Update button hover state
            button.draw(screen)          # Draw the button

        # --- Draw Informational Text ---
        # Define starting position for info text (below the buttons)
        info_y = 380 # Adjust this value based on button layout
        line = 0     # Line counter for vertical spacing

        # Helper function to draw a line of text in the UI panel
        def draw_info(text, val=""):
             nonlocal line # Allow modification of the outer 'line' variable
             # Format text with optional value
             full_text = f"{text}: {val}" if val != "" else text
             # Render the text surface
             surf = self.font_small.render(full_text, True, WHITE)
             # Blit the text onto the screen at the calculated position
             screen.blit(surf, (UI_PANEL_X + BUTTON_PADDING, info_y + line * UI_LINE_HEIGHT))
             line += 1 # Increment line counter for next text item

        # Draw various simulation statistics
        draw_info("FPS", f"{int(clock.get_fps())}")
        draw_info("Speed", f"{self.speed_multiplier:.1f}x")
        draw_info("Agents", f"{len(self.agents)}")
        # draw_info("Resources", f"{len(self.resources)}") # Less dynamic count
        total_res_quantity = sum(r.quantity for r in self.resources)
        draw_info("World Res Qty", f"{total_res_quantity}")
        draw_info("Base Storage", f"{self.total_base_resources}")
        draw_info("Brush", f"{self.paint_brush}")
        line +=1 # Add extra spacing before selected agent info

        # --- Selected Agent Information ---
        draw_info("Selected Agent:")
        if self.selected_agent and self.selected_agent in self.agents:
            agent = self.selected_agent
            # Display detailed info about the selected agent
            draw_info("  ID", agent.id)
            state_str = agent.state
            # Add action type if in working state
            if agent.state == STATE_WORKING and agent.action_type:
                 state_str += f" ({agent.action_type})"
            draw_info("  State", state_str)
            draw_info("  Energy", f"{agent.energy:.1f} / {MAX_ENERGY}")
            draw_info("  Hunger", f"{agent.hunger:.1f} / {MAX_HUNGER}")
            draw_info("  Carrying", agent.carrying_resource)
            # Display target information if agent has one
            if agent.target_resource:
                 draw_info("  Target", f"Res {agent.target_resource.id} @ {agent.target_resource.grid_pos}")
            elif agent.destination_grid_pos:
                 draw_info("  Target", f"Grid {agent.destination_grid_pos}")
            elif agent.target_pos_world: # Moving towards intermediate node
                 grid_target = world_to_grid(agent.target_pos_world.x, agent.target_pos_world.y)
                 draw_info("  Target", f"Node {grid_target}")

        else:
             draw_info("  None") # No agent selected

        # --- Draw Pause Indicator ---
        if game_state['paused']:
             pause_surf = self.font_medium.render("PAUSED", True, YELLOW)
             # Position indicator near the bottom center of the UI panel
             pause_rect = pause_surf.get_rect(centerx=UI_PANEL_X + UI_PANEL_WIDTH // 2, y=SCREEN_HEIGHT - 40)
             screen.blit(pause_surf, pause_rect)


# --- Save/Load Functions ---
SAVE_FILENAME = "agent_sim_save_v2.pkl" # Default filename for saving/loading

def save_simulation(environment, filename=SAVE_FILENAME):
    """Saves the current state of the simulation environment to a file using pickle."""
    print(f"Attempting to save simulation to {filename}...")
    # --- Prepare environment for pickling ---
    # 1. Backup non-serializable attributes (pygame fonts, surfaces, etc.)
    font_small_backup = environment.font_small
    font_medium_backup = environment.font_medium
    selected_id_backup = environment.selected_agent.id if environment.selected_agent else None
    buttons_backup = environment.buttons # Buttons contain font references, cannot be pickled directly

    # 2. Remove or replace non-serializable attributes
    environment.font_small = None
    environment.font_medium = None
    environment.selected_agent = None # Save ID instead of the object reference
    environment.buttons = []          # Buttons will be recreated on load
    environment.quadtree = None       # Quadtree will be rebuilt on load

    # 3. Create data structure to save
    save_data = {
        'environment_state': environment.__dict__, # Save the core attributes dictionary
        'selected_agent_id': selected_id_backup   # Save the ID of the selected agent
    }

    # 4. Perform pickling
    try:
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Simulation saved successfully.")
    except Exception as e:
        print(f"Error saving simulation: {e}")
    finally:
        # --- Restore transient attributes to the original environment object ---
        # This is crucial if the simulation continues running after saving
        environment.font_small = font_small_backup
        environment.font_medium = font_medium_backup
        environment.buttons = buttons_backup # Restore original button list
        # Find agent object from restored ID (if simulation continues)
        if selected_id_backup is not None:
            for agent in environment.agents:
                if agent.id == selected_id_backup:
                    environment.selected_agent = agent
                    break
        # Rebuild quadtree if simulation continues
        if hasattr(environment, 'sim_width'): # Check if env object still valid
             environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
             environment.rebuild_quadtree()


def load_simulation(filename=SAVE_FILENAME):
    """Loads a simulation state from a file and returns a new environment object."""
    print(f"Attempting to load simulation from {filename}...")
    # Check if the save file exists
    if not os.path.exists(filename):
        print(f"Error: Save file '{filename}' not found.")
        return None, None # Return None for environment and selected ID

    try:
        # Load the data from the pickle file
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # --- Recreate the environment object ---
        # 1. Create a new Environment instance (provide dummy fonts initially)
        env_state = save_data['environment_state']
        loaded_env = Environment(
            env_state['width'],
            env_state['height'],
            env_state['sim_width'],
            env_state['sim_height'],
            None, None # Dummy fonts, will be replaced later in main loop
        )

        # 2. Update the new environment's dictionary with the loaded state
        # This overwrites default values with the saved ones
        loaded_env.__dict__.update(env_state)

        # 3. Get the ID of the agent that was selected when saved
        selected_id = save_data['selected_agent_id']

        print(f"Simulation loaded successfully.")
        # Return the newly created environment and the selected agent's ID
        return loaded_env, selected_id
    except Exception as e:
        print(f"Error loading simulation: {e}")
        return None, None # Return None on error


# --- Main Game Loop ---
def main():
    # --- Pygame Initialization ---
    pygame.init()
    pygame.font.init()

    # Set up display window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Agent Simulation")

    # Clock for controlling frame rate and delta time
    clock = pygame.time.Clock()

    # Load fonts (these are transient, need to be re-assigned after load)
    try:
        font_small = pygame.font.SysFont(None, UI_FONT_SIZE_SMALL)
        font_medium = pygame.font.SysFont(None, UI_FONT_SIZE_MEDIUM)
    except Exception as e:
        print(f"Error loading system font: {e}. Using default.")
        font_small = pygame.font.Font(None, UI_FONT_SIZE_SMALL) # Pygame default font
        font_medium = pygame.font.Font(None, UI_FONT_SIZE_MEDIUM)

    # --- Game State Dictionary ---
    # Stores persistent state flags toggled by user actions
    game_state = {
        'paused': False,       # Is the simulation paused?
        'draw_grid': True,       # Should the terrain grid be drawn?
        'draw_quadtree': False,    # Should the quadtree boundaries be drawn (debug)?
        'is_painting': False     # Is the left mouse button held down for painting?
    }

    # --- Create Environment ---
    environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT, SIM_WIDTH, SIM_HEIGHT, font_small, font_medium)

    # --- Initial Population Setup ---
    # Place the base (e.g., near left-center)
    base_x, base_y = SIM_WIDTH // 3, SIM_HEIGHT // 2
    environment.set_base(Base(base_x, base_y))
    if not environment.base:
        print("Critical Error: Failed to place base. Exiting.")
        return # Cannot run simulation without a base

    # Ensure the area around the base is initially walkable (clear potential obstacles)
    if environment.base.grid_pos:
         gx, gy = environment.base.grid_pos
         for dx in range(-1, 2): # 3x3 area around base center
             for dy in range(-1, 2):
                  clear_x, clear_y = gx + dx, gy + dy
                  # Check bounds before modifying grid
                  if 0 <= clear_x < GRID_WIDTH and 0 <= clear_y < GRID_HEIGHT:
                      # Set terrain to plains, ensuring it's not accidentally an obstacle
                      environment.grid[clear_y][clear_x] = TERRAIN_PLAINS

    # Add some initial obstacles using the line function
    environment.create_obstacle_line(SIM_WIDTH * 0.6, SIM_HEIGHT * 0.1, SIM_WIDTH * 0.7, SIM_HEIGHT * 0.9)
    environment.create_obstacle_line(SIM_WIDTH * 0.1, SIM_HEIGHT * 0.8, SIM_WIDTH * 0.5, SIM_HEIGHT * 0.7)
    # Add some random terrain for variety (optional)
    # for _ in range(50): environment.paint_terrain((random.randint(0, SIM_WIDTH), random.randint(0, SIM_HEIGHT)), brush_size=3) # Example random patches

    # Add initial Resources and Agents
    for _ in range(40): environment.create_random_entity('resource')
    for _ in range(60): environment.create_random_entity('agent')

    # --- Main Loop ---
    running = True
    while running:
        # --- Delta Time Calculation ---
        # Get time elapsed since last frame in milliseconds, convert to seconds
        base_dt = clock.tick(FPS) / 1000.0
        # Clamp dt to prevent large jumps if frame rate drops significantly
        dt = min(base_dt, 0.1)
        # Use dt for updates only if not paused
        effective_dt = dt if not game_state['paused'] else 0

        # --- Event Handling ---
        mouse_pos = pygame.mouse.get_pos() # Get current mouse position
        # Define rectangle for the simulation area
        sim_area_rect = pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT)
        # Check if mouse is within the simulation area
        mouse_in_sim = sim_area_rect.collidepoint(mouse_pos)

        # Process all events in the queue
        for event in pygame.event.get():
            # Handle Quit event (closing window)
            if event.type == pygame.QUIT:
                running = False

            # --- Keyboard Input ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # Exit on ESC
                    running = False
                # --- Save/Load Hotkeys ---
                elif event.key == pygame.K_F5: # Save Simulation
                     save_simulation(environment)
                     # Note: Transients are restored within save_simulation if needed
                     # But ensure UI/Quadtree are functional if game continues immediately
                     # (This is handled by save_simulation's finally block and next frame's rebuild)

                elif event.key == pygame.K_F9: # Load Simulation
                    loaded_env, selected_id = load_simulation()
                    if loaded_env:
                        environment = loaded_env # Replace current environment with loaded one
                        # --- Restore Transient State after Loading ---
                        # 1. Assign the correct font objects
                        environment.font_small = font_small
                        environment.font_medium = font_medium
                        # 2. Recreate the UI buttons using the loaded state and fonts
                        environment._setup_ui_buttons()
                        # 3. Recreate and rebuild the quadtree
                        environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
                        environment.rebuild_quadtree() # Populate with loaded entities
                        # 4. Find and re-select the agent based on the loaded ID
                        environment.selected_agent = None
                        if selected_id is not None:
                             for agent in environment.agents:
                                 if agent.id == selected_id:
                                     environment.selected_agent = agent
                                     break
                        print("Transient states (fonts, UI, quadtree, selection) restored after load.")
                # --- Debug Keys (Add entities) ---
                elif event.key == pygame.K_a: # Add random agent
                    environment.create_random_entity('agent')
                elif event.key == pygame.K_r: # Add random resource
                    environment.create_random_entity('resource')

            # --- Mouse Input ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Mouse Button Click
                    # --- Check UI Panel First ---
                    if mouse_pos[0] >= UI_PANEL_X:
                        # Handle click in UI panel, check if save/load was triggered
                        action_result = environment.handle_ui_click(mouse_pos, game_state)
                        if action_result == "save":
                            save_simulation(environment)
                            # Transients restored in save_simulation's finally block
                        elif action_result == "load":
                            # Trigger load using F9 logic (cleaner to keep load logic in one place)
                            print("Load requested via button. Press F9 to confirm load.")
                            # Alternatively, directly call load logic here:
                            # loaded_env, selected_id = load_simulation()
                            # ... [Restoration code as in F9 handler] ...
                            pass # Keeping F9 as primary load trigger for simplicity

                    # --- Check Simulation Area ---
                    elif mouse_in_sim:
                        # If select tool is active, try to select an agent
                        if environment.paint_brush == BRUSH_SELECT:
                            environment.select_agent_at(mouse_pos)
                        # Otherwise, start painting terrain/walls
                        else:
                            game_state['is_painting'] = True
                            environment.paint_terrain(mouse_pos) # Paint single cell on click

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1: # Left Mouse Button Release
                      # Stop painting when mouse button is released
                      game_state['is_painting'] = False

            elif event.type == pygame.MOUSEMOTION:
                 # If painting is active (mouse button held down) and mouse is in sim area
                 if game_state['is_painting'] and mouse_in_sim:
                      # Continue painting terrain/walls while dragging
                      if environment.paint_brush != BRUSH_SELECT:
                           environment.paint_terrain(mouse_pos)

        # --- Simulation Updates (only if not paused) ---
        if not game_state['paused']:
            # 1. Rebuild Quadtree: Update spatial partitioning with current positions
            environment.rebuild_quadtree()
            # 2. Update Resources: Handle regeneration
            environment.update_resources(dt) # Pass base dt, multiplier applied inside
            # 3. Update Agents: Handle movement, state changes, needs, actions
            environment.update_agents(dt) # Pass base dt, multiplier applied inside

        # --- Drawing ---
        screen.fill(BLACK) # Clear the entire screen (or a dark background color)

        # Draw the Simulation Area contents
        environment.draw_sim_area(screen, game_state['draw_grid'], game_state['draw_quadtree'])

        # Draw the UI Panel contents
        environment.draw_ui(screen, clock, game_state)

        # Update the display to show the newly drawn frame
        pygame.display.flip()

    # --- Cleanup ---
    pygame.quit()

# --- Run the simulation ---
if __name__ == '__main__':
    main()
