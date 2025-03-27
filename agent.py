import pygame
import random
import math
from constants import * # Import all constants
from utils import world_to_grid, grid_to_world_center
from pathfinding import astar_pathfinding
# Need to import Resource type for type checking, avoid circular by importing module
import entities

class Agent:
    """Represents an autonomous agent in the simulation."""
    _id_counter = 0
    def __init__(self, x, y, environment):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.environment = environment # Reference to the environment
        self.pos = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
        self.max_speed = AGENT_SPEED
        self.max_force = AGENT_MAX_FORCE

        # --- Role Assignment ---
        self.role = random.choice(AGENT_ROLES)
        if self.role == ROLE_COLLECTOR:
            self.color = BLUE
        elif self.role == ROLE_BUILDER:
            self.color = ORANGE
        else: # Default
            self.color = PURPLE # Should not happen with current choices

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
        self.capacity = AGENT_CAPACITY

        # Clamp initial position
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))
        # Rectangle for drawing and quadtree
        self.rect = pygame.Rect(self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE, AGENT_SIZE * 2, AGENT_SIZE * 2)
        self.update_rect()
        self.entity_type = "AGENT" # For identification

    def update_rect(self):
        """Updates the agent's rectangle based on its current position."""
        self.rect.center = (int(self.pos.x), int(self.pos.y))

    def apply_force(self, force):
        """Applies a force vector to the agent's velocity."""
        # This method is conceptually useful but force is applied directly in update loop
        self.velocity += force

    def seek(self, target_pos):
        """Calculates a steering force to move towards a target position."""
        if target_pos is None: return pygame.Vector2(0, 0)
        desired = target_pos - self.pos
        dist_sq = desired.length_squared()
        # Use a slightly smaller arrival radius than half a grid cell
        arrival_threshold_sq = (GRID_CELL_SIZE * 0.4)**2
        if dist_sq < arrival_threshold_sq:
            return pygame.Vector2(0, 0) # Arrived

        if dist_sq > 0 : # Avoid normalizing zero vector
            desired.normalize_ip()
            desired *= self.max_speed
        else:
            return pygame.Vector2(0, 0) # Already exactly at target?

        steer = desired - self.velocity
        # Limit the steering force
        if steer.length_squared() > self.max_force * self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def separation(self, neighbors):
        """Calculates a steering force to avoid crowding nearby agents."""
        steer = pygame.Vector2(0, 0)
        count = 0
        desired_separation = (AGENT_SIZE * 2) * 1.5 # Desired minimum distance

        for other in neighbors:
            if other is self: continue
            dist_sq = self.pos.distance_squared_to(other.pos)
            # Check if within separation radius
            if 0 < dist_sq < desired_separation * desired_separation:
                diff = self.pos - other.pos
                # Weight by inverse distance squared (stronger repulsion closer)
                diff *= (1.0 / (dist_sq + 0.001)) # Add epsilon to avoid division by zero
                steer += diff
                count += 1

        if count > 0:
            steer /= count # Average steering vector
            if steer.length_squared() > 0:
                steer.normalize_ip()
                steer *= self.max_speed # Desired velocity is away from neighbors
                steer -= self.velocity # Calculate steering force
                if steer.length_squared() > self.max_force * self.max_force:
                    steer.scale_to_length(self.max_force) # Limit force
        return steer

    def _find_path(self, target_grid_pos):
        """Attempts to find a path using A*."""
        start_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        # Reset current path state
        self.current_path = []
        self.path_index = 0
        self.target_pos_world = None
        self.destination_grid_pos = target_grid_pos # Store the final target

        # Check validity
        if start_grid_pos is None or target_grid_pos is None: return False
        if start_grid_pos == target_grid_pos: return True # Already there

        # Access grid via environment reference for pathfinding
        path = astar_pathfinding(self.environment.grid, start_grid_pos, target_grid_pos)

        if path:
            self.current_path = path
            self.path_index = 0
            # Set first node as immediate target
            next_grid_node = self.current_path[self.path_index]
            self.target_pos_world = pygame.Vector2(grid_to_world_center(next_grid_node[0], next_grid_node[1]))
            return True
        else:
            # Pathfinding failed
            self.destination_grid_pos = None # Clear destination if no path found
            return False

    def update(self, dt, nearby_agents):
        """Main update logic for the agent."""
        # --- Needs Update (Factoring in Day/Night) ---
        current_energy_decay = ENERGY_DECAY_RATE_NIGHT if self.environment.is_night else ENERGY_DECAY_RATE_DAY
        self.energy -= current_energy_decay * dt
        self.hunger += HUNGER_INCREASE_RATE * dt
        self.energy = max(0, self.energy)
        self.hunger = min(MAX_HUNGER, self.hunger)

        # Check for death
        if self.energy <= 0:
            self.environment.log_event(f"Agent {self.id} ({self.role}) died of exhaustion.")
            self.environment.remove_agent(self)
            return # Stop processing this agent

        # --- Steering Forces ---
        seek_force = pygame.Vector2(0, 0)
        separation_force = self.separation(nearby_agents) * 1.5 # Separation weight

        # --- High Priority Need Overrides ---
        needs_override = False
        is_moving_for_needs = self.state == STATE_MOVING_TO_BASE and (self.action_type == ACTION_EATING or self.action_type == ACTION_RESTING)

        # Check Hunger
        if self.hunger >= HIGH_HUNGER_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             if self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 if self._find_path(self.environment.base.grid_pos):
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_EATING
                     self._release_resource_target()
                     needs_override = True

        # Check Energy (only if hunger didn't trigger)
        elif self.energy <= LOW_ENERGY_THRESHOLD and self.state != STATE_WORKING and not is_moving_for_needs:
             if not needs_override and self.environment.base and self.state != STATE_MOVING_TO_BASE:
                 if self._find_path(self.environment.base.grid_pos):
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_RESTING
                     self._release_resource_target()
                     needs_override = True

        # --- State Machine & Path Following ---
        path_completed_this_frame = False

        # Follow path if one exists
        if self.current_path:
            if self.target_pos_world:
                 dist_to_node_sq = self.pos.distance_squared_to(self.target_pos_world)
                 arrival_threshold_sq = (GRID_CELL_SIZE * 0.5)**2 # Wider arrival radius

                 if dist_to_node_sq < arrival_threshold_sq:
                     self.path_index += 1 # Move to next node
                     if self.path_index < len(self.current_path):
                         # Update target to next node
                         next_node_grid = self.current_path[self.path_index]
                         self.target_pos_world = pygame.Vector2(grid_to_world_center(next_node_grid[0], next_node_grid[1]))
                     else: # Reached end of path
                         self.current_path = []
                         self.target_pos_world = None
                         path_completed_this_frame = True
                         # Snap to final destination center
                         if self.destination_grid_pos:
                             self.pos = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))
                         self.velocity *= 0.1 # Dampen velocity on arrival
            else: # Path exists but no target node? Clear path.
                self.current_path = []
                self.path_index = 0

        # Determine seek target (current node or final destination)
        seek_target = self.target_pos_world
        if path_completed_this_frame and self.destination_grid_pos:
             seek_target = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))

        # Apply seek force towards the target
        if seek_target:
             seek_force = self.seek(seek_target)

        # --- State Logic (only if needs didn't override) ---
        if not needs_override:
            # --- IDLE State: Decide action based on Role ---
            if self.state == STATE_IDLE:
                # Common Priority: Return resources if carrying any
                if self.carrying_resource > 0:
                    if self.environment.base and self._find_path(self.environment.base.grid_pos):
                         self.state = STATE_MOVING_TO_BASE
                         self.action_type = ACTION_RETURNING
                # --- Role Specific Decisions ---
                # BUILDER Role: Prioritize building
                elif self.role == ROLE_BUILDER:
                    # 1. Try to Build if possible
                    build_possible = self.carrying_resource >= BUILD_COST and self.energy > LOW_ENERGY_THRESHOLD + 15
                    if build_possible:
                        build_spot = self._find_build_spot()
                        if build_spot and self._find_path(build_spot):
                            self.state = STATE_MOVING_TO_BUILD
                        # If cannot find build spot, collect if not full
                        elif self.carrying_resource < self.capacity:
                             resource_target = self._find_best_available_resource(max_search_radius=200) # Builder might search closer
                             if resource_target and self._find_path(resource_target.grid_pos):
                                 self.state = STATE_MOVING_TO_RESOURCE
                                 self.target_resource = resource_target
                                 self.environment.mark_resource_targeted(resource_target, self)
                    # 2. If cannot build (low res/energy), collect if not full
                    elif self.carrying_resource < self.capacity:
                        resource_target = self._find_best_available_resource()
                        if resource_target and self._find_path(resource_target.grid_pos):
                            self.state = STATE_MOVING_TO_RESOURCE
                            self.target_resource = resource_target
                            self.environment.mark_resource_targeted(resource_target, self)
                    # Else: Builder remains idle if full and cannot build

                # COLLECTOR Role: Prioritize collecting
                elif self.role == ROLE_COLLECTOR:
                    # 1. Collect if not full
                    if self.carrying_resource < self.capacity:
                        resource_target = self._find_best_available_resource()
                        if resource_target and self._find_path(resource_target.grid_pos):
                            self.state = STATE_MOVING_TO_RESOURCE
                            self.target_resource = resource_target
                            self.environment.mark_resource_targeted(resource_target, self)
                    # Else: Collector remains idle if full

                # Default Role (if any) or fallback: Collect
                else:
                    if self.carrying_resource < self.capacity:
                        resource_target = self._find_best_available_resource()
                        if resource_target and self._find_path(resource_target.grid_pos):
                            self.state = STATE_MOVING_TO_RESOURCE
                            self.target_resource = resource_target
                            self.environment.mark_resource_targeted(resource_target, self)

            # --- Moving States: Transition upon path completion ---
            elif self.state == STATE_MOVING_TO_RESOURCE and path_completed_this_frame:
                # Check if resource still valid and targeted by self
                if self.target_resource and self.target_resource.quantity > 0 and self.environment.is_resource_targeted_by(self.target_resource, self):
                    self.state = STATE_WORKING
                    self.action_type = ACTION_COLLECTING
                    self.action_timer = 1.0 + random.uniform(-0.2, 0.2) # Time per unit
                else: # Resource gone or taken
                    self._release_resource_target()
                    self.state = STATE_IDLE

            elif self.state == STATE_MOVING_TO_BUILD and path_completed_this_frame:
                current_grid_pos = world_to_grid(self.pos.x, self.pos.y)
                build_target = self.destination_grid_pos
                # Check if arrived at intended spot and it's still buildable
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
                 # Check reason for going to base
                 if self.action_type == ACTION_RETURNING:
                     self.environment.add_base_resources(self.carrying_resource)
                     self.environment.log_event(f"Agent {self.id} ({self.role}) deposited {self.carrying_resource} res.")
                     self.carrying_resource = 0
                     self.action_type = None
                     self.state = STATE_IDLE # Re-evaluate
                 elif self.action_type == ACTION_EATING:
                     self.state = STATE_WORKING # Start eating timer
                     self.action_timer = EAT_TIME
                     # action_type remains ACTION_EATING
                 elif self.action_type == ACTION_RESTING:
                     self.state = STATE_WORKING # Start resting timer
                     self.action_timer = REST_TIME
                     # action_type remains ACTION_RESTING
                 else: # Arrived for unknown reason
                     self.action_type = None
                     self.state = STATE_IDLE

            # --- WORKING State: Perform timed action ---
            elif self.state == STATE_WORKING:
                self.action_timer -= dt
                if self.action_timer <= 0:
                    action_done = False
                    # Perform action based on type
                    if self.action_type == ACTION_COLLECTING:
                        if self.target_resource and self.target_resource.quantity > 0:
                            collected = self.target_resource.collect(1)
                            if collected > 0:
                                self.carrying_resource += collected
                                # Stop if full or resource depleted
                                if self.carrying_resource >= self.capacity or self.target_resource.quantity == 0:
                                    if self.target_resource.quantity == 0:
                                        self.environment.log_event(f"Res {self.target_resource.id} depleted.")
                                    self._release_resource_target()
                                    action_done = True
                                else: # Reset timer to collect next unit
                                    self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                            else: # Collect failed (race condition?)
                                self._release_resource_target()
                                action_done = True
                        else: # Resource vanished while collecting
                            self._release_resource_target()
                            action_done = True

                    elif self.action_type == ACTION_BUILDING:
                         build_pos = self.destination_grid_pos
                         if build_pos and self.carrying_resource >= BUILD_COST:
                              # Attempt build via environment
                              build_success = self.environment.build_wall(build_pos[0], build_pos[1], self)
                              if build_success:
                                  self.carrying_resource -= BUILD_COST
                                  self.environment.log_event(f"Agent {self.id} ({self.role}) built wall at {build_pos}.")
                              # else: Failed or not enough resources
                         self.destination_grid_pos = None # Clear build target
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
        # Apply acceleration (F/m * dt, assume m=1)
        self.velocity += total_force * dt

        # Limit velocity
        vel_mag_sq = self.velocity.length_squared()
        if vel_mag_sq > self.max_speed * self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        # Apply damping based on state
        elif self.state == STATE_WORKING:
             self.velocity *= 0.05 # Almost stop
        elif self.state == STATE_IDLE and not self.target_pos_world and not self.current_path:
             self.velocity *= 0.85 # Slow down gradually if idle with no immediate target

        # Update position
        self.pos += self.velocity * dt

        # --- Boundary Constraints ---
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))

        # Update Rect for Quadtree and drawing
        self.update_rect()


    def _release_resource_target(self):
        """Informs the environment that this agent is no longer targeting its current resource."""
        if self.target_resource:
            self.environment.mark_resource_available(self.target_resource)
            self.target_resource = None

    def _find_best_available_resource(self, max_search_radius=300):
        """Finds the 'best' available and untargeted resource within a radius."""
        best_score = -float('inf')
        best_resource = None
        # Use Quadtree to find potential resources
        potential_resources = self.environment.quadtree.query_radius(self.pos, max_search_radius)

        # Filter: Resource type, quantity > 0, not targeted
        available_resources = [
            res for res in potential_resources
            if isinstance(res, entities.Resource) and res.quantity > 0 and res.id not in self.environment.targeted_resources
        ]
        if not available_resources: return None

        # Scoring weights
        weight_quantity = 1.5 # Higher quantity better
        weight_distance = -0.05 # Closer distance better

        for resource in available_resources:
            dist_sq = self.pos.distance_squared_to(resource.pos)
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1
            score = (weight_quantity * resource.quantity) + (weight_distance * dist)
            if score > best_score:
                 best_score = score
                 best_resource = resource
        return best_resource

    def _find_build_spot(self):
        """Finds a valid, adjacent grid cell to build on."""
        my_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        if my_grid_pos is None: return None

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4 directions
        random.shuffle(neighbors) # Check in random order
        for dx, dy in neighbors:
            check_x, check_y = my_grid_pos[0] + dx, my_grid_pos[1] + dy
            # Use environment's check for buildability
            if self.environment.is_buildable(check_x, check_y):
                 return (check_x, check_y)
        return None # No suitable spot found

    def draw(self, screen):
         """Draws the agent on the screen."""
         # Don't draw if outside sim area
         if self.pos.x >= SIM_WIDTH or self.pos.y >= SIM_HEIGHT or self.pos.x < 0 or self.pos.y < 0:
             return

         center_tuple = (int(self.pos.x), int(self.pos.y))
         # Base color is determined by role in __init__
         pygame.draw.circle(screen, self.color, center_tuple, AGENT_SIZE)

         # --- State Indicator (Inner Circle) ---
         state_color = WHITE # Default for IDLE
         # Determine color based on state and action
         if self.state == STATE_MOVING_TO_RESOURCE: state_color = CYAN
         elif self.state == STATE_MOVING_TO_BUILD: state_color = ORANGE # Maybe use Grey? Builder color is Orange.
         elif self.state == STATE_MOVING_TO_BASE:
             if self.action_type == ACTION_RETURNING: state_color = YELLOW
             elif self.action_type == ACTION_EATING: state_color = RED
             elif self.action_type == ACTION_RESTING: state_color = GREEN
             else: state_color = PURPLE # Generic move to base
         elif self.state == STATE_WORKING:
             if self.action_type == ACTION_COLLECTING: state_color = CYAN # Changed collecting color
             elif self.action_type == ACTION_BUILDING: state_color = GREY
             elif self.action_type == ACTION_EATING: state_color = RED
             elif self.action_type == ACTION_RESTING: state_color = GREEN
             else: state_color = DARK_GREY # Generic working
         # Draw inner circle
         pygame.draw.circle(screen, state_color, center_tuple, max(1, AGENT_SIZE // 2))

         # --- Draw Needs Bars below agent ---
         bar_width = AGENT_SIZE * 2
         bar_height = 3
         bar_padding = 1
         bar_x = self.pos.x - bar_width / 2
         bar_y_energy = self.pos.y + AGENT_SIZE + 2
         bar_y_hunger = bar_y_energy + bar_height + bar_padding

         # Energy Bar (Green)
         energy_ratio = self.energy / MAX_ENERGY
         pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_energy, bar_width, bar_height))
         pygame.draw.rect(screen, GREEN, (bar_x, bar_y_energy, bar_width * energy_ratio, bar_height))

         # Hunger Bar (Red)
         hunger_ratio = self.hunger / MAX_HUNGER
         pygame.draw.rect(screen, DARK_GREY, (bar_x, bar_y_hunger, bar_width, bar_height))
         pygame.draw.rect(screen, RED, (bar_x, bar_y_hunger, bar_width * hunger_ratio, bar_height))