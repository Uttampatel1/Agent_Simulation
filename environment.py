import pygame
import random
import math # For day/night calc
from collections import deque

# --- Specific Imports from constants ---
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_WIDTH, SIM_HEIGHT, UI_PANEL_WIDTH,
    GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, AGENT_SIZE,
    RESOURCE_SIZE, BASE_SIZE, BUILD_COST, BUILD_TIME, WATER_SOURCE_SIZE,
    MAX_ENERGY, MAX_HUNGER, LOW_ENERGY_THRESHOLD, HIGH_HUNGER_THRESHOLD,
    ENERGY_DECAY_RATE_DAY, ENERGY_DECAY_RATE_NIGHT, HUNGER_INCREASE_RATE,
    EAT_AMOUNT, REST_AMOUNT, EAT_TIME, REST_TIME, AGENT_CAPACITY,
    MAX_THIRST, THIRST_INCREASE_RATE, HIGH_THIRST_THRESHOLD, DRINK_AMOUNT, DRINK_TIME, # Thirst
    MAX_SLEEPINESS, SLEEPINESS_INCREASE_RATE_AWAKE, SLEEPINESS_DECREASE_RATE_ASLEEP, # Sleep
    HIGH_SLEEPINESS_THRESHOLD, SLEEP_ENERGY_REGEN_RATE,
    MAX_SOCIAL, SOCIAL_DECAY_RATE, LOW_SOCIAL_THRESHOLD, SOCIAL_GAIN_RATE, # Social
    SOCIALIZE_TIME, SOCIAL_INTERACTION_RANGE,
    MAX_BOREDOM, BOREDOM_INCREASE_RATE_WORKING, BOREDOM_INCREASE_RATE_IDLE, # Boredom
    HIGH_BOREDOM_THRESHOLD, BOREDOM_DECREASE_WHILE_LEISURE, LEISURE_TIME,
    MAX_FEAR, FEAR_INCREASE_PER_SECOND_NEAR_THREAT, FEAR_DECAY_RATE, # Fear
    HIGH_FEAR_THRESHOLD, FLEE_SPEED_MULTIPLIER, THREAT_RADIUS,
    DAY_NIGHT_CYCLE_DURATION, DAY_DURATION_RATIO, MAX_NIGHT_ALPHA,
    FPS, MAX_EVENT_LOG_MESSAGES,
    WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, CYAN, GREY,
    DARK_GREY, LIGHT_GREY, OBSTACLE_COLOR, BASE_COLOR, NIGHT_OVERLAY_COLOR,
    WATER_COLOR, THREAT_COLOR, # New Colors
    TERRAIN_EMPTY, TERRAIN_PLAINS, TERRAIN_GRASS, TERRAIN_MUD, TERRAIN_WALL,
    TERRAIN_WATER, OBSTACLE_COST, TERRAIN_TYPES, TERRAIN_COLORS, # Water Terrain
    BRUSH_SELECT, BRUSH_WALL, BRUSH_PLAINS, BRUSH_GRASS, BRUSH_MUD, BRUSH_WATER, # Water Brush
    BRUSH_CLEAR, BRUSH_SPAWN_AGENT, BRUSH_SPAWN_RESOURCE, BRUSH_SPAWN_WATER, # Water Spawn
    BRUSH_SPAWN_THREAT, BRUSH_DELETE, PAINT_BRUSHES, SPAWN_BRUSHES, # Threat Spawn
    MIN_BRUSH_SIZE, MAX_BRUSH_SIZE, ROLE_COLLECTOR, ROLE_BUILDER
)

from utils import world_to_grid, grid_to_world_center, clamp
from quadtree import Quadtree
# Import ALL entity types now
from entities import Obstacle, Base, Resource, WaterSource, Threat
from agent import Agent

class Environment:
    """Manages the simulation state, entities, grid, day/night cycle, and updates."""
    def __init__(self, width, height, sim_width, sim_height):
        self.width = width; self.height = height
        self.sim_width = sim_width; self.sim_height = sim_height

        # Entity lists
        self.agents = []
        self.resources = []
        self.water_sources = [] # New list for water sources
        self.threats = []       # New list for threats
        self.obstacles = []
        self.base = None
        self.total_base_resources = 0
        self.targeted_resources = {}

        # World Representation
        self.grid = [[TERRAIN_PLAINS for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.quadtree = Quadtree(0, pygame.Rect(0, 0, sim_width, sim_height))

        # UI related state
        self.selected_agent = None; self.speed_multiplier = 1.0
        self.paint_brush = BRUSH_SELECT; self.brush_size = 1

        # Event Log
        self.event_log = deque(maxlen=MAX_EVENT_LOG_MESSAGES * 2)

        # Day/Night Cycle
        self.world_time = 0.0; self.is_night = False; self.night_alpha = 0

    def log_event(self, message): self.event_log.append(message)
    def get_event_log(self): return list(self.event_log)

    def paint_terrain(self, mouse_pos):
        """Paints terrain/walls/water onto the grid."""
        grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
        if grid_pos is None: return

        gx, gy = grid_pos
        terrain_to_paint = None # Use terrain type/cost directly
        is_clearing = False
        current_brush = self.paint_brush

        # Determine terrain type based on brush
        if current_brush == BRUSH_WALL: terrain_to_paint = TERRAIN_WALL
        elif current_brush == BRUSH_WATER: terrain_to_paint = TERRAIN_WATER # New
        elif current_brush == BRUSH_CLEAR:
            terrain_to_paint = TERRAIN_PLAINS; is_clearing = True
        elif current_brush == BRUSH_PLAINS: terrain_to_paint = TERRAIN_PLAINS
        elif current_brush == BRUSH_GRASS: terrain_to_paint = TERRAIN_GRASS
        elif current_brush == BRUSH_MUD: terrain_to_paint = TERRAIN_MUD
        else: return # Not a painting brush

        if terrain_to_paint is not None:
             offset = self.brush_size // 2
             for dx in range(-offset, offset + self.brush_size % 2):
                 for dy in range(-offset, offset + self.brush_size % 2):
                      paint_x, paint_y = gx + dx, gy + dy
                      if 0 <= paint_x < GRID_WIDTH and 0 <= paint_y < GRID_HEIGHT:
                           # Check conflicts before painting
                           is_base = self.base and self.base.grid_pos == (paint_x, paint_y)
                           is_res = any(r.grid_pos == (paint_x, paint_y) for r in self.resources)
                           is_water_src = any(w.grid_pos == (paint_x, paint_y) for w in self.water_sources)

                           # Allow painting if not base/res/water_src (unless clearing)
                           can_paint = not (is_base or is_res or is_water_src)

                           if can_paint:
                               current_terrain = self.grid[paint_y][paint_x]
                               # If clearing, also remove potential obstacle object
                               if is_clearing and current_terrain == OBSTACLE_COST:
                                   self.obstacles = [o for o in self.obstacles if o.grid_pos != (paint_x, paint_y)]
                                   # Don't clear water terrain back to plains unless intended?
                                   # Maybe CLEAR only affects WALLS? Let's assume it clears walls/water.
                                   self.grid[paint_y][paint_x] = terrain_to_paint

                               elif not is_clearing:
                                   # Apply terrain change
                                   self.grid[paint_y][paint_x] = terrain_to_paint
                                   # Add obstacle object if painting wall
                                   if terrain_to_paint == TERRAIN_WALL:
                                       if not any(o.grid_pos == (paint_x, paint_y) for o in self.obstacles):
                                           self.obstacles.append(Obstacle(paint_x, paint_y))
                                   # If painting over a wall/obstacle, remove it from list
                                   elif current_terrain == OBSTACLE_COST:
                                        self.obstacles = [o for o in self.obstacles if o.grid_pos != (paint_x, paint_y)]


    def is_buildable(self, grid_x, grid_y):
         """Checks if a wall can be built (not on obstacles, water, mud, entities)."""
         if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             terrain_cost = self.grid[grid_y][grid_x]
             # Cannot build on impassable, water, or mud
             if terrain_cost not in [OBSTACLE_COST, TERRAIN_WALL, TERRAIN_WATER, TERRAIN_MUD]:
                  if not (self.base and self.base.grid_pos == (grid_x, grid_y)):
                      if not any(r.grid_pos == (grid_x, grid_y) for r in self.resources):
                          if not any(w.grid_pos == (grid_x, grid_y) for w in self.water_sources):
                             return True
         return False

    # --- Entity Management ---
    def add_agent(self, agent): self.agents.append(agent)
    def remove_agent(self, agent_to_remove):
        agent_to_remove._release_resource_target(); self.agents = [a for a in self.agents if a.id != agent_to_remove.id]
        if self.selected_agent and self.selected_agent.id == agent_to_remove.id: self.selected_agent = None
    def add_resource(self, resource):
        if resource.grid_pos is None: return
        gx, gy = resource.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST and self.grid[gy][gx] != TERRAIN_WATER: # Don't place on water terrain either
            self.resources.append(resource); self.log_event(f"Res {resource.id} spawned @ {resource.grid_pos}.")
        else: self.log_event(f"Failed spawn resource @ obstacle/water {resource.grid_pos}.")
    def add_water_source(self, water_src): # New method
        if water_src.grid_pos is None: return
        gx, gy = water_src.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST: # Can place on any walkable terrain
             # Optionally change underlying grid to water terrain?
             # self.grid[gy][gx] = TERRAIN_WATER
             self.water_sources.append(water_src); self.log_event(f"Water Src {water_src.id} spawned @ {water_src.grid_pos}.")
        else: self.log_event(f"Failed spawn water src @ obstacle {water_src.grid_pos}.")
    def add_threat(self, threat): # New method
        self.threats.append(threat); self.log_event(f"Threat {threat.id} spawned near ({int(threat.pos.x)}, {int(threat.pos.y)}).")

    def add_obstacle(self, grid_x, grid_y):
        # ... (checks for base, res, water source) ...
        if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
             if self.grid[grid_y][grid_x] != OBSTACLE_COST:
                  if self.base and self.base.grid_pos == (grid_x, grid_y): return False
                  if any(r.grid_pos == (grid_x, grid_y) for r in self.resources): return False
                  if any(w.grid_pos == (grid_x, grid_y) for w in self.water_sources): return False # Check water sources
                  self.grid[grid_y][grid_x] = OBSTACLE_COST
                  if not any(o.grid_pos == (grid_x, grid_y) for o in self.obstacles): self.obstacles.append(Obstacle(grid_x, grid_y))
                  return True
        return False

    def delete_entity_at(self, grid_x, grid_y):
        """Removes Agents, Resources, Water Sources, Threats, Walls at grid coord."""
        if not (0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT): return False
        deleted_something = False
        cell_rect = pygame.Rect(grid_x*GRID_CELL_SIZE, grid_y*GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)

        # Agents (collision based)
        agents_to_delete = [a for a in self.agents if cell_rect.colliderect(a.rect)]
        for agent in agents_to_delete: self.log_event(f"Agent {agent.id} deleted."); self.remove_agent(agent); deleted_something = True
        # Resources (grid based)
        res_at_pos = [r for r in self.resources if r.grid_pos == (grid_x, grid_y)]
        for res in res_at_pos:
             if res.id in self.targeted_resources: self.mark_resource_available(res)
             self.log_event(f"Resource {res.id} deleted."); self.resources.remove(res); deleted_something = True
        # Water Sources (grid based)
        water_at_pos = [w for w in self.water_sources if w.grid_pos == (grid_x, grid_y)]
        for water in water_at_pos: self.log_event(f"Water Src {water.id} deleted."); self.water_sources.remove(water); deleted_something = True
        # Threats (collision based - needs rect or check distance)
        threats_to_delete = [t for t in self.threats if cell_rect.collidepoint(t.pos)]
        for threat in threats_to_delete: self.log_event(f"Threat {threat.id} deleted."); self.threats.remove(threat); deleted_something = True
        # Walls/Obstacles/Water Terrain (reset grid to plains)
        if self.grid[grid_y][grid_x] == OBSTACLE_COST or self.grid[grid_y][grid_x] == TERRAIN_WATER:
            if self.grid[grid_y][grid_x] == OBSTACLE_COST: self.log_event(f"Wall cleared @ {(grid_x, grid_y)}.")
            else: self.log_event(f"Water terrain cleared @ {(grid_x, grid_y)}.")
            self.grid[grid_y][grid_x] = TERRAIN_PLAINS
            self.obstacles = [o for o in self.obstacles if o.grid_pos != (grid_x, grid_y)]
            deleted_something = True

        return deleted_something

    # set_base, add_base_resources, build_wall (same as before)
    def set_base(self, base):
        if base.grid_pos is None: return
        gx, gy = base.grid_pos
        if self.grid[gy][gx] != OBSTACLE_COST: self.base = base; self.log_event(f"Base set @ {base.grid_pos}.")
        else: self.log_event(f"Warning: Cannot set base @ occupied {base.grid_pos}")
    def add_base_resources(self, amount): self.total_base_resources += amount
    def build_wall(self, grid_x, grid_y, builder_agent):
         if self.is_buildable(grid_x, grid_y):
             self.grid[grid_y][grid_x] = TERRAIN_WALL
             if not any(o.grid_pos == (grid_x, grid_y) for o in self.obstacles): self.obstacles.append(Obstacle(grid_x, grid_y))
             return True
         return False

    # --- Creation Methods (Water/Threat added) ---
    def create_agent_at(self, world_x, world_y):
        grid_pos = world_to_grid(world_x, world_y)
        if grid_pos and self.grid[grid_pos[1]][grid_pos[0]] != OBSTACLE_COST:
            new_agent = Agent(world_x, world_y, self); self.add_agent(new_agent)
            self.log_event(f"Agent {new_agent.id} ({new_agent.role}) spawned @ {grid_pos}."); return True
        self.log_event(f"Failed spawn agent @ {grid_pos} (obstructed)."); return False
    def create_resource_at(self, world_x, world_y):
        grid_pos = world_to_grid(world_x, world_y)
        if grid_pos and self.grid[grid_pos[1]][grid_pos[0]] != OBSTACLE_COST and self.grid[grid_pos[1]][grid_pos[0]] != TERRAIN_WATER:
            try: wx, wy = grid_to_world_center(*grid_pos); self.add_resource(Resource(wx, wy)); return True
            except ValueError as e: self.log_event(f"Error spawn resource: {e}"); return False
        self.log_event(f"Failed spawn resource @ {grid_pos} (obstructed/water)."); return False
    def create_water_source_at(self, world_x, world_y): # New
        grid_pos = world_to_grid(world_x, world_y)
        if grid_pos and self.grid[grid_pos[1]][grid_pos[0]] != OBSTACLE_COST:
            try: wx, wy = grid_to_world_center(*grid_pos); self.add_water_source(WaterSource(wx, wy)); return True
            except ValueError as e: self.log_event(f"Error spawn water src: {e}"); return False
        self.log_event(f"Failed spawn water src @ {grid_pos} (obstructed)."); return False
    def create_threat_at(self, world_x, world_y): # New
        if 0 <= world_x < SIM_WIDTH and 0 <= world_y < SIM_HEIGHT:
            self.add_threat(Threat(world_x, world_y)); return True
        self.log_event(f"Failed spawn threat outside sim area."); return False

    def create_random_entity(self, entity_type):
        attempts = 0; max_attempts = 100
        while attempts < max_attempts:
            attempts += 1; gx = random.randint(1, GRID_WIDTH-2); gy = random.randint(1, GRID_HEIGHT-2)
            if self.grid[gy][gx] != OBSTACLE_COST:
                 wx, wy = grid_to_world_center(gx, gy); success = False
                 if entity_type == 'agent': success = self.create_agent_at(wx, wy)
                 elif entity_type == 'resource': success = self.create_resource_at(wx, wy)
                 elif entity_type == 'water': success = self.create_water_source_at(wx, wy) # Handle 'water' type
                 elif entity_type == 'threat': success = self.create_threat_at(wx, wy) # Handle 'threat' type
                 if success: return
        self.log_event(f"Warning: Could not find spot for random {entity_type}.")
    def create_obstacle_line(self, x1, y1, x2, y2):
        # ... (same as before, using add_obstacle) ...
        x1=clamp(x1,0,SIM_WIDTH-1); y1=clamp(y1,0,SIM_HEIGHT-1); x2=clamp(x2,0,SIM_WIDTH-1); y2=clamp(y2,0,SIM_HEIGHT-1)
        start_node=world_to_grid(x1,y1); end_node=world_to_grid(x2,y2);
        if start_node is None or end_node is None: return
        gx1,gy1=start_node; gx2,gy2=end_node; dx=abs(gx2-gx1); dy=-abs(gy2-gy1); sx=1 if gx1<gx2 else -1; sy=1 if gy1<gy2 else -1; err=dx+dy; count=0; max_count=GRID_WIDTH*GRID_HEIGHT
        while count < max_count:
             self.add_obstacle(gx1, gy1) # add_obstacle handles checks
             if gx1==gx2 and gy1==gy2: break
             e2=2*err;
             if e2>=dy:
                 if gx1==gx2: break; err+=dy; gx1+=sx
             if e2<=dx:
                 if gy1==gy2: break; err+=dx; gy1+=sy
             count+=1

    # --- Getters for Agent AI ---
    def get_water_sources_near(self, pos, radius):
        """Returns water sources within radius using quadtree."""
        potential = self.quadtree.query_radius(pos, radius)
        return [w for w in potential if isinstance(w, WaterSource)]
    def get_threats_near(self, pos, radius):
        """Returns threats within radius (simple list check or quadtree)."""
        # If threats list is small, simple check is fine
        nearby_threats = []
        radius_sq = radius * radius
        for threat in self.threats:
            if pos.distance_squared_to(threat.pos) <= radius_sq:
                nearby_threats.append(threat)
        return nearby_threats
        # Alternatively, if threats are in quadtree:
        # potential = self.quadtree.query_radius(pos, radius)
        # return [t for t in potential if isinstance(t, Threat)]
    def get_social_location(self):
        """Returns the primary location for socializing (the base)."""
        return self.base.pos if self.base else None

    # mark/release/is resource targeted (same as before)
    def mark_resource_targeted(self, resource, agent):
        if resource and resource.id not in self.targeted_resources: self.targeted_resources[resource.id] = agent.id
    def mark_resource_available(self, resource):
        if resource and resource.id in self.targeted_resources: del self.targeted_resources[resource.id]
    def is_resource_targeted_by(self, resource, agent):
        return self.targeted_resources.get(resource.id) == agent.id
    # select_agent_at (same as before)
    def select_agent_at(self, mouse_pos):
        # ... (same as before) ...
        if mouse_pos[0] >= self.sim_width or mouse_pos[1] >= self.sim_height or mouse_pos[0] < 0 or mouse_pos[1] < 0: self.selected_agent = None; return
        self.selected_agent = None; search_radius = AGENT_SIZE * 3
        nearby_entities = self.quadtree.query_radius(mouse_pos, search_radius)
        nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]
        min_dist_sq = search_radius * search_radius; closest_agent = None; click_vec = pygame.Vector2(mouse_pos)
        for agent in nearby_agents:
             dist_sq = agent.pos.distance_squared_to(click_vec)
             if dist_sq < min_dist_sq: min_dist_sq = dist_sq; closest_agent = agent
        self.selected_agent = closest_agent
        if self.selected_agent: self.log_event(f"Selected Agent {self.selected_agent.id} ({self.selected_agent.role})")

    # _update_day_night_cycle (same as before)
    def _update_day_night_cycle(self, effective_dt):
        if effective_dt <= 0: return
        self.world_time += effective_dt; time_in_cycle = self.world_time % DAY_NIGHT_CYCLE_DURATION
        day_end_time = DAY_NIGHT_CYCLE_DURATION * DAY_DURATION_RATIO; was_night = self.is_night
        self.is_night = time_in_cycle > day_end_time
        if self.is_night != was_night: self.log_event("Night has fallen." if self.is_night else "Day has broken.")
        transition_time = DAY_NIGHT_CYCLE_DURATION * 0.1
        if self.is_night:
            time_into_night = time_in_cycle - day_end_time; time_before_day = DAY_NIGHT_CYCLE_DURATION - time_in_cycle
            if time_into_night < transition_time: self.night_alpha = int(MAX_NIGHT_ALPHA * (time_into_night / transition_time))
            elif time_before_day < transition_time: self.night_alpha = int(MAX_NIGHT_ALPHA * (time_before_day / transition_time))
            else: self.night_alpha = MAX_NIGHT_ALPHA
        else:
            time_into_day = time_in_cycle; time_before_night = day_end_time - time_in_cycle
            if time_into_day < transition_time: self.night_alpha = int(MAX_NIGHT_ALPHA * (1.0 - (time_into_day / transition_time)))
            elif time_before_night < transition_time: self.night_alpha = int(MAX_NIGHT_ALPHA * (1.0 - (time_before_night / transition_time)))
            else: self.night_alpha = 0
        self.night_alpha = clamp(self.night_alpha, 0, MAX_NIGHT_ALPHA)

    def update(self, dt):
        """Main update loop for the environment."""
        effective_dt = dt * self.speed_multiplier
        self._update_day_night_cycle(effective_dt)
        self.rebuild_quadtree()
        # Update entities (pass effective_dt)
        for entity_list in [self.resources, self.water_sources, self.threats, self.agents]:
             # Use slicing [:] to iterate over a copy if entities might be removed during update
             for entity in entity_list[:]:
                 if hasattr(entity, 'update'):
                      # Agents need nearby agents list
                      if isinstance(entity, Agent) and entity in self.agents: # Check if still exists
                           nearby_radius = (AGENT_SIZE * 2) * 3
                           nearby_entities = self.quadtree.query_radius(entity.pos, nearby_radius)
                           nearby_agents = [e for e in nearby_entities if isinstance(e, Agent)]
                           entity.update(effective_dt, nearby_agents)
                      elif entity in self.agents: # Handle case where entity removed mid-loop
                           pass
                      else: # For other updatable entities (Resources)
                           entity.update(effective_dt)

    def rebuild_quadtree(self):
        """Clears and rebuilds quadtree with relevant entities."""
        self.quadtree.clear()
        # Add entities that need spatial querying
        entities_to_insert = self.agents + self.resources + self.water_sources # Add threats if needed for queries
        for entity in entities_to_insert:
            if hasattr(entity, 'rect') and self.quadtree.bounds.colliderect(entity.rect):
                 if hasattr(entity, 'update_rect'): entity.update_rect()
                 elif hasattr(entity, 'pos') and isinstance(entity.pos, pygame.Vector2):
                      entity.rect.center = (int(entity.pos.x), int(entity.pos.y))
                 # Final check for valid position before inserting
                 if 0 <= entity.rect.centerx < self.sim_width and 0 <= entity.rect.centery < self.sim_height:
                     self.quadtree.insert(entity)

    # --- Drawing Methods ---
    def draw_sim_area(self, screen, draw_grid_flag, draw_quadtree_flag):
        # 1. Background
        if draw_grid_flag: self.draw_grid(screen)
        else: screen.fill(BLACK, (0, 0, self.sim_width, self.sim_height))
        # 2. Static/Semi-static Entities
        if self.base: self.base.draw(screen)
        for water in self.water_sources: water.draw(screen)
        # 3. Dynamic Entities
        for res in self.resources: res.draw(screen, res.id in self.targeted_resources)
        for agent in self.agents: agent.draw(screen)
        for threat in self.threats: threat.draw(screen) # Draw threats (debug)
        # 4. Debug
        if draw_quadtree_flag: self.quadtree.draw(screen)
        # 5. Selection Highlight
        if self.selected_agent and self.selected_agent in self.agents:
             center = (int(self.selected_agent.pos.x), int(self.selected_agent.pos.y))
             if 0 <= center[0] < self.sim_width and 0 <= center[1] < self.sim_height:
                 pygame.draw.circle(screen, WHITE, center, AGENT_SIZE + 3, 1)
        # 6. Night Overlay
        if self.night_alpha > 0:
            night_surface = pygame.Surface((self.sim_width, self.sim_height), pygame.SRCALPHA)
            night_surface.fill((*NIGHT_OVERLAY_COLOR, int(self.night_alpha)))
            screen.blit(night_surface, (0, 0))

    def draw_grid(self, screen):
        # ... (same as before - uses TERRAIN_COLORS which now includes water) ...
        sim_rect = pygame.Rect(0, 0, self.sim_width, self.sim_height)
        for gy in range(GRID_HEIGHT):
            for gx in range(GRID_WIDTH):
                terrain_cost = self.grid[gy][gx]; color = TERRAIN_COLORS.get(terrain_cost, GREY)
                rect = pygame.Rect(gx*GRID_CELL_SIZE, gy*GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
                if sim_rect.colliderect(rect): pygame.draw.rect(screen, color, rect)