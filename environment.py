import pygame
import random
import math # For day/night calc
from collections import deque
from constants import *
from utils import world_to_grid, grid_to_world_center, clamp
from quadtree import Quadtree
from entities import Obstacle, Base, Resource
from agent import Agent

class Environment:
    """Manages the simulation state, entities, grid, day/night cycle, and updates."""
    def __init__(self, width, height, sim_width, sim_height):
        self.width = width
        self.height = height
        self.sim_width = sim_width
        self.sim_height = sim_height

        # Entity lists and state
        self.agents = []
        self.resources = []
        self.obstacles = [] # Mainly for referencing placed obstacles if needed
        self.base = None
        self.total_base_resources = 0
        self.targeted_resources = {} # {resource_id: agent_id}

        # World Representation
        self.grid = [[TERRAIN_PLAINS for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.quadtree = Quadtree(0, pygame.Rect(0, 0, sim_width, sim_height))

        # UI related state
        self.selected_agent = None
        self.speed_multiplier = 1.0
        self.paint_brush = BRUSH_SELECT
        self.brush_size = 1

        # Event Log
        self.event_log = deque(maxlen=MAX_EVENT_LOG_MESSAGES * 2) # Store more than displayed

        # --- Day/Night Cycle ---
        self.world_time = 0.0 # Total elapsed simulation time in seconds
        self.is_night = False
        self.night_alpha = 0 # Transparency of night overlay (0-MAX_NIGHT_ALPHA)

        # Reset class counters (optional, careful with loading)
        Agent._id_counter = 0
        Resource._id_counter = 0


    def log_event(self, message):
        """Adds a message to the event log queue."""
        # Could add timestamp here if desired
        self.event_log.append(message)

    def get_event_log(self):
        """Returns the current list of log messages."""
        return list(self.event_log)

    def paint_terrain(self, mouse_pos):
        """Paints terrain/walls onto the grid based on current brush and size."""
        grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
        if grid_pos is None: return # Outside sim area

        gx, gy = grid_pos
        terrain_cost_to_paint = None
        is_clearing = False
        current_brush = self.paint_brush

        # Determine terrain type based on brush
        if current_brush == BRUSH_WALL: terrain_cost_to_paint = TERRAIN_WALL
        elif current_brush == BRUSH_CLEAR:
            terrain_cost_to_paint = TERRAIN_PLAINS # Clearing sets back to plains
            is_clearing = True
        elif current_brush == BRUSH_PLAINS: terrain_cost_to_paint = TERRAIN_PLAINS
        elif current_brush == BRUSH_GRASS: terrain_cost_to_paint = TERRAIN_GRASS
        elif current_brush == BRUSH_MUD: terrain_cost_to_paint = TERRAIN_MUD
        else: return # Not a painting brush

        if terrain_cost_to_paint is not None:
             # Apply paint in square area based on brush size
             offset = self.brush_size // 2
             # Adjust range for even/odd sizes to center correctly
             for dx in range(-offset, offset + self.brush_size % 2):
                 for dy in range(-offset, offset + self.brush_size % 2):
                      paint_x, paint_y = gx + dx, gy + dy
                      # Ensure paint location is within grid bounds
                      if 0 <= paint_x < GRID_WIDTH and 0 <= paint_y < GRID_HEIGHT:
                           # Check if painting over base or resource
                           is_base = self.base and self.base.grid_pos == (paint_x, paint_y)
                           is_res = any(r.grid_pos == (paint_x, paint_y) for r in self.resources)

                           # Allow painting if not base/resource
                           if not is_base and not is_res:
                               # If clearing, also remove from obstacles list
                               if is_clearing and self.grid[paint_y][paint_x] == OBSTACLE_COST:
                                   self.obstacles = [o for o in self.obstacles if o.grid_pos != (paint_x, paint_y)]

                               # Apply terrain change to grid
                               self.grid[paint_y][paint_x] = terrain_cost_to_paint

                               # If painting a wall, add to obstacles list (avoid duplicates)
                               if terrain_cost_to_paint == TERRAIN_WALL:
                                    if not any(o.grid_pos == (paint_x, paint_y) for o in self.obstacles):
                                        self.obstacles.append(Obstacle(paint_x, paint_y))


    def is_buildable(self, grid_x, grid_y):
         """Checks if a grid cell is suitable for building a wall on."""
         if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             terrain_cost = self.grid[grid_y][grid_x]
             # Buildable on non-impassable, non-mud terrain
             if terrain_cost not in [OBSTACLE_COST, TERRAIN_WALL, TERRAIN_MUD]:
                  # Check not base or resource spot
                  if not (self.base and self.base.grid_pos == (grid_x, grid_y)):
                      if not any(r.grid_pos == (grid_x, grid_y) for r in self.resources):
                          return True
         return False

    def add_agent(self, agent):
        """Adds an agent to the simulation."""
        self.agents.append(agent)
        # Quadtree insertion happens during rebuild

    def remove_agent(self, agent_to_remove):
        """Removes an agent from the simulation."""
        agent_to_remove._release_resource_target() # Ensure resource is freed
        self.agents = [a for a in self.agents if a.id != agent_to_remove.id]
        # Deselect if the removed agent was selected
        if self.selected_agent and self.selected_agent.id == agent_to_remove.id:
             self.selected_agent = None

    def add_resource(self, resource):
        """Adds a resource node if the location is valid."""
        if resource.grid_pos is None: return # Should be caught earlier
        gx, gy = resource.grid_pos
        # Only add if the cell is not an obstacle
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.resources.append(resource)
            self.log_event(f"Resource {resource.id} spawned at {resource.grid_pos}.")
        else:
             self.log_event(f"Failed to spawn resource at obstacle {resource.grid_pos}.")

    def add_obstacle(self, grid_x, grid_y):
        """Adds an obstacle by updating the grid cost and obstacle list."""
        if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             # Check if already an obstacle
             if self.grid[grid_y][grid_x] != OBSTACLE_COST:
                  # Prevent placing on base or existing resources
                  if self.base and self.base.grid_pos == (grid_x, grid_y): return False
                  if any(r.grid_pos == (grid_x, grid_y) for r in self.resources): return False

                  # Update grid and obstacle list
                  self.grid[grid_y][grid_x] = OBSTACLE_COST
                  if not any(o.grid_pos == (grid_x, grid_y) for o in self.obstacles):
                      self.obstacles.append(Obstacle(grid_x, grid_y))
                  # self.log_event(f"Obstacle placed at {(grid_x, grid_y)}.") # Can be noisy
                  return True
             return False # Already an obstacle
        return False # Out of bounds

    def delete_entity_at(self, grid_x, grid_y):
        """Removes entities (Agent, Resource, Wall/Obstacle) at a grid coordinate."""
        if not (0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT): return False

        deleted_something = False
        # Delete Agents at this grid cell
        # Need world pos check as agent rect might span cells
        cell_rect = pygame.Rect(grid_x * GRID_CELL_SIZE, grid_y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
        agents_to_delete = [a for a in self.agents if cell_rect.colliderect(a.rect)] # Check rect collision
        if not agents_to_delete: # Fallback check center if rect check missed
             agents_to_delete = [a for a in self.agents if world_to_grid(a.pos.x, a.pos.y) == (grid_x, grid_y)]

        for agent in agents_to_delete:
             self.log_event(f"Agent {agent.id} deleted.")
             self.remove_agent(agent)
             deleted_something = True

        # Delete Resources at this grid cell
        res_at_pos = [r for r in self.resources if r.grid_pos == (grid_x, grid_y)]
        for res in res_at_pos:
             if res.id in self.targeted_resources: # Make available first
                 self.mark_resource_available(res)
             self.log_event(f"Resource {res.id} deleted.")
             self.resources.remove(res)
             deleted_something = True

        # Delete Obstacles/Walls (set terrain back to plains)
        if self.grid[grid_y][grid_x] == OBSTACLE_COST:
            self.grid[grid_y][grid_x] = TERRAIN_PLAINS # Reset terrain
            self.obstacles = [o for o in self.obstacles if o.grid_pos != (grid_x, grid_y)] # Remove from list
            self.log_event(f"Wall/Obstacle cleared at {(grid_x, grid_y)}.")
            deleted_something = True

        # Base cannot be deleted this way

        return deleted_something

    def set_base(self, base):
        """Sets the simulation's base, checking validity."""
        if base.grid_pos is None: return
        gx, gy = base.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST:
            self.base = base
            self.log_event(f"Base set at {base.grid_pos}.")
        else:
            self.log_event(f"Warning: Cannot set base at occupied grid cell {base.grid_pos}")

    def add_base_resources(self, amount):
        """Increases the total resources stored at the base."""
        self.total_base_resources += amount

    def build_wall(self, grid_x, grid_y, builder_agent):
         """Called by an agent to build a wall."""
         if self.is_buildable(grid_x, grid_y):
             self.grid[grid_y][grid_x] = TERRAIN_WALL # Use wall cost (impassable)
             # Add to obstacles list for consistency if needed elsewhere
             if not any(o.grid_pos == (grid_x, grid_y) for o in self.obstacles):
                 self.obstacles.append(Obstacle(grid_x, grid_y))
             return True
         return False

    def create_agent_at(self, world_x, world_y):
        """Creates an agent if the location is valid (not obstructed)."""
        grid_pos = world_to_grid(world_x, world_y)
        if grid_pos and self.grid[grid_pos[1]][grid_pos[0]] != OBSTACLE_COST:
            new_agent = Agent(world_x, world_y, self)
            self.add_agent(new_agent)
            self.log_event(f"Agent {new_agent.id} ({new_agent.role}) spawned at {grid_pos}.")
            return True
        self.log_event(f"Failed to spawn agent at {grid_pos} (obstructed).")
        return False

    def create_resource_at(self, world_x, world_y):
        """Creates a resource if the location is valid (not obstructed)."""
        grid_pos = world_to_grid(world_x, world_y)
        if grid_pos and self.grid[grid_pos[1]][grid_pos[0]] != OBSTACLE_COST:
            try:
                # Ensure resource is created within bounds (should be okay if grid_pos valid)
                wx_center, wy_center = grid_to_world_center(grid_pos[0], grid_pos[1])
                self.add_resource(Resource(wx_center, wy_center))
                return True
            except ValueError as e:
                self.log_event(f"Error spawning resource: {e}")
                return False
        self.log_event(f"Failed to spawn resource at {grid_pos} (obstructed).")
        return False

    def create_random_entity(self, entity_type):
        """Attempts to create an agent or resource at a random valid location."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            attempts += 1
            # Choose random grid cell (avoiding edges)
            gx = random.randint(1, GRID_WIDTH - 2)
            gy = random.randint(1, GRID_HEIGHT - 2)
            # Check if cell is not an obstacle
            if self.grid[gy][gx] != OBSTACLE_COST:
                 wx, wy = grid_to_world_center(gx, gy)
                 success = False
                 if entity_type == 'agent':
                      success = self.create_agent_at(wx, wy)
                 elif entity_type == 'resource':
                      success = self.create_resource_at(wx, wy)
                 if success: return # Exit if successful
        # Log failure if max attempts reached
        self.log_event(f"Warning: Could not find spot for random {entity_type} after {max_attempts} attempts.")

    def create_obstacle_line(self, x1, y1, x2, y2):
        """Creates a line of obstacles on the grid using Bresenham's."""
        # Clamp coords and convert to grid
        x1 = clamp(x1, 0, SIM_WIDTH - 1); y1 = clamp(y1, 0, SIM_HEIGHT - 1)
        x2 = clamp(x2, 0, SIM_WIDTH - 1); y2 = clamp(y2, 0, SIM_HEIGHT - 1)
        start_node = world_to_grid(x1, y1)
        end_node = world_to_grid(x2, y2)
        if start_node is None or end_node is None: return

        # Bresenham's algorithm
        gx1, gy1 = start_node; gx2, gy2 = end_node
        dx = abs(gx2 - gx1); dy = -abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1; sy = 1 if gy1 < gy2 else -1
        err = dx + dy
        count = 0; max_count = GRID_WIDTH * GRID_HEIGHT # Safety break

        while count < max_count:
             self.add_obstacle(gx1, gy1) # Use internal method to add obstacle
             if gx1 == gx2 and gy1 == gy2: break # Reached end
             e2 = 2 * err
             if e2 >= dy: # Step X ?
                 if gx1 == gx2: break # Avoid loop on vertical lines
                 err += dy; gx1 += sx
             if e2 <= dx: # Step Y ?
                 if gy1 == gy2: break # Avoid loop on horizontal lines
                 err += dx; gy1 += sy
             count += 1
        # Log if loop terminated unexpectedly
        # if count == max_count: print("Warning: Obstacle line exceeded max count.")

    # --- Resource Targeting ---
    def mark_resource_targeted(self, resource, agent):
        if resource and resource.id not in self.targeted_resources:
            self.targeted_resources[resource.id] = agent.id
    def mark_resource_available(self, resource):
        if resource and resource.id in self.targeted_resources:
            del self.targeted_resources[resource.id]
    def is_resource_targeted_by(self, resource, agent):
        return self.targeted_resources.get(resource.id) == agent.id

    # --- Agent Selection ---
    def select_agent_at(self, mouse_pos):
        """Selects the agent closest to the mouse click position."""
        # Ensure selection only happens within sim area
        if mouse_pos[0] >= self.sim_width or mouse_pos[1] >= self.sim_height or mouse_pos[0] < 0 or mouse_pos[1] < 0:
             self.selected_agent = None
             return

        self.selected_agent = None # Deselect previous
        search_radius = AGENT_SIZE * 3 # Increase search radius slightly
        # Query quadtree for nearby agents
        nearby_entities = self.quadtree.query_radius(mouse_pos, search_radius)
        nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]

        # Find the closest one within the radius
        min_dist_sq = search_radius * search_radius
        closest_agent = None
        click_vec = pygame.Vector2(mouse_pos)
        for agent in nearby_agents:
             dist_sq = agent.pos.distance_squared_to(click_vec)
             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_agent = agent
        self.selected_agent = closest_agent
        if self.selected_agent:
            self.log_event(f"Selected Agent {self.selected_agent.id} ({self.selected_agent.role})")


    def _update_day_night_cycle(self, effective_dt):
        """Updates the world time and day/night state based on effective delta time."""
        if effective_dt <= 0: return # Don't update time if paused

        self.world_time += effective_dt
        time_in_cycle = self.world_time % DAY_NIGHT_CYCLE_DURATION
        day_end_time = DAY_NIGHT_CYCLE_DURATION * DAY_DURATION_RATIO

        was_night = self.is_night
        self.is_night = time_in_cycle > day_end_time

        # Log transition
        if self.is_night != was_night:
            self.log_event("Night has fallen." if self.is_night else "Day has broken.")

        # --- Calculate night overlay alpha for smooth transition ---
        transition_time = DAY_NIGHT_CYCLE_DURATION * 0.1 # 10% of cycle duration for fade in/out

        if self.is_night:
            time_into_night = time_in_cycle - day_end_time
            time_before_day = DAY_NIGHT_CYCLE_DURATION - time_in_cycle
            if time_into_night < transition_time: # Fading in
                self.night_alpha = int(MAX_NIGHT_ALPHA * (time_into_night / transition_time))
            elif time_before_day < transition_time: # Fading out before day
                self.night_alpha = int(MAX_NIGHT_ALPHA * (time_before_day / transition_time))
            else: # Full night
                self.night_alpha = MAX_NIGHT_ALPHA
        else: # Day time
            time_into_day = time_in_cycle
            time_before_night = day_end_time - time_in_cycle
            if time_into_day < transition_time: # Fading out after night
                 self.night_alpha = int(MAX_NIGHT_ALPHA * (1.0 - (time_into_day / transition_time)))
            elif time_before_night < transition_time: # Fading in before night
                self.night_alpha = int(MAX_NIGHT_ALPHA * (1.0 - (time_before_night / transition_time)))
            else: # Full day
                self.night_alpha = 0

        # Clamp alpha value just in case
        self.night_alpha = clamp(self.night_alpha, 0, MAX_NIGHT_ALPHA)

    def update(self, dt):
        """Main update loop for the environment. Applies speed multiplier."""
        # Calculate effective delta time for this frame based on speed multiplier
        # Time progresses even if visually paused if we use dt here,
        # but simulation logic uses effective_dt. Let's use effective_dt for time too.
        effective_dt = dt * self.speed_multiplier

        # 0. Update Day/Night Cycle
        self._update_day_night_cycle(effective_dt)

        # 1. Rebuild Quadtree (Essential for accurate neighbor finding)
        self.rebuild_quadtree()

        # 2. Update Resources (Handles regeneration)
        for resource in self.resources:
            resource.update(effective_dt) # Pass effective dt

        # 3. Update Agents (AI, Movement, Needs)
        agents_to_update = self.agents[:] # Iterate over a copy for safe removal
        for agent in agents_to_update:
              if agent in self.agents: # Check if agent still exists
                   # Find nearby agents using quadtree
                   nearby_radius = (AGENT_SIZE * 2) * 3
                   nearby_entities = self.quadtree.query_radius(agent.pos, nearby_radius)
                   nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]
                   # Pass effective dt to agent's update method
                   agent.update(effective_dt, nearby_agents)


    def rebuild_quadtree(self):
        """Clears and rebuilds the quadtree with current entity positions."""
        self.quadtree.clear()
        # Include agents and resources in the quadtree
        entities_to_insert = self.agents + self.resources
        for entity in entities_to_insert:
            # Ensure entity has a rect and is roughly within bounds
            if hasattr(entity, 'rect') and self.quadtree.bounds.colliderect(entity.rect):
                 # Update rect position before inserting if applicable
                 if hasattr(entity, 'update_rect'):
                     entity.update_rect()
                 elif hasattr(entity, 'pos') and hasattr(entity.rect, 'center'): # Fallback for static entities
                     entity.rect.center = (int(entity.pos.x), int(entity.pos.y))

                 # Final check: only insert if center is within sim bounds
                 if 0 <= entity.rect.centerx < self.sim_width and 0 <= entity.rect.centery < self.sim_height:
                    self.quadtree.insert(entity)

    # --- Drawing Methods ---
    def draw_sim_area(self, screen, draw_grid_flag, draw_quadtree_flag):
        """Draws all elements within the simulation area."""
        # 1. Draw background (Grid or Black)
        if draw_grid_flag:
            self.draw_grid(screen)
        else:
            sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
            screen.fill(BLACK, sim_rect)

        # 2. Draw static elements (Base)
        if self.base:
            self.base.draw(screen)

        # 3. Draw dynamic elements (Resources, Agents)
        # Obstacles are implicitly drawn by draw_grid now if using TERRAIN_WALL
        for res in self.resources:
            res.draw(screen, res.id in self.targeted_resources) # Indicate if targeted
        for agent in self.agents:
            agent.draw(screen)

        # 4. Draw debug elements (Quadtree)
        if draw_quadtree_flag:
            self.quadtree.draw(screen)

        # 5. Draw highlight for selected agent
        if self.selected_agent and self.selected_agent in self.agents:
             # Ensure highlight is within sim bounds
             center_x = int(self.selected_agent.pos.x)
             center_y = int(self.selected_agent.pos.y)
             if 0 <= center_x < self.sim_width and 0 <= center_y < self.sim_height:
                 pygame.draw.circle(screen, WHITE, (center_x, center_y), AGENT_SIZE + 3, 1) # White outline

        # 6. Draw Night Overlay (on top of everything else in sim area)
        if self.night_alpha > 0:
            # Create a surface with per-pixel alpha
            night_surface = pygame.Surface((self.sim_width, self.sim_height), pygame.SRCALPHA)
            # Fill with the night color and calculated alpha
            night_surface.fill((*NIGHT_OVERLAY_COLOR, self.night_alpha))
            # Blit the overlay onto the main screen
            screen.blit(night_surface, (0, 0))

    def draw_grid(self, screen):
        """Draws grid cells with terrain colors."""
        sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
        # Iterate through grid cells
        for gy in range(GRID_HEIGHT):
            for gx in range(GRID_WIDTH):
                terrain_cost = self.grid[gy][gx]
                color = TERRAIN_COLORS.get(terrain_cost, GREY) # Get color from cost
                rect = pygame.Rect(gx * GRID_CELL_SIZE, gy * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
                # Draw rect if it's within the sim area
                if sim_rect.colliderect(rect): # Basic check
                    pygame.draw.rect(screen, color, rect)
                    # Optional: Draw grid lines
                    pygame.draw.rect(screen, DARK_GREY, rect, 1)