import pygame
import random
import math
from constants import *
from utils import world_to_grid, grid_to_world_center
from pathfinding import astar_pathfinding
import entities # Use module import

class Agent:
    """Represents an autonomous agent with complex needs."""
    _id_counter = 0
    def __init__(self, x, y, environment):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.environment = environment
        self.pos = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
        self.max_speed = AGENT_SPEED
        self.max_force = AGENT_MAX_FORCE

        # Role & Appearance
        self.role = random.choice(AGENT_ROLES)
        self.color = ORANGE if self.role == ROLE_BUILDER else BLUE

        # State Machine
        self.state = STATE_IDLE
        self.action_type = None
        self.action_timer = 0

        # Pathfinding & Targeting
        self.target_resource = None # Resource entity target
        self.target_water = None    # WaterSource entity target
        self.target_agent = None    # Agent entity target (for social)
        self.target_pos_world = None # Current path node world pos
        self.destination_grid_pos = None # Final grid destination
        self.current_path = []
        self.path_index = 0

        # --- Needs ---
        self.energy = MAX_ENERGY * random.uniform(0.7, 1.0) # Start slightly lower maybe
        self.hunger = MAX_HUNGER * random.uniform(0.0, 0.3)
        self.thirst = MAX_THIRST * random.uniform(0.0, 0.3)
        self.sleepiness = MAX_SLEEPINESS * random.uniform(0.0, 0.2)
        self.social = MAX_SOCIAL * random.uniform(0.5, 1.0) # Start reasonably social
        self.boredom = MAX_BOREDOM * random.uniform(0.0, 0.2)
        self.fear = 0 # Start with no fear

        # Inventory
        self.carrying_resource = 0
        self.capacity = AGENT_CAPACITY

        # Position clamping and rect init
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))
        self.rect = pygame.Rect(self.pos.x - AGENT_SIZE, self.pos.y - AGENT_SIZE, AGENT_SIZE * 2, AGENT_SIZE * 2)
        self.update_rect()
        self.entity_type = "AGENT"

        # Internal state flags
        self._current_threat = None # Store the threat causing fear


    def update_rect(self): self.rect.center = (int(self.pos.x), int(self.pos.y))
    def apply_force(self, force): self.velocity += force
    def seek(self, target_pos, arrival_radius_multiplier=0.4):
        """Calculates steering force towards target."""
        if target_pos is None: return pygame.Vector2(0, 0)
        desired = target_pos - self.pos
        dist_sq = desired.length_squared()
        arrival_threshold_sq = (GRID_CELL_SIZE * arrival_radius_multiplier)**2
        if dist_sq < arrival_threshold_sq: return pygame.Vector2(0, 0)
        if dist_sq > 0 : desired.normalize_ip(); desired *= self.max_speed
        else: return pygame.Vector2(0, 0)
        steer = desired - self.velocity
        if steer.length_squared() > self.max_force * self.max_force: steer.scale_to_length(self.max_force)
        return steer

    def flee(self, threat_pos):
        """Calculates steering force away from a threat."""
        if threat_pos is None: return pygame.Vector2(0,0)
        desired = (self.pos - threat_pos) # Vector pointing away from threat
        if desired.length_squared() > 0:
            desired.normalize_ip()
            # Flee faster
            desired *= self.max_speed * FLEE_SPEED_MULTIPLIER
            steer = desired - self.velocity
            # Apply strong force to change direction quickly
            if steer.length_squared() > self.max_force * self.max_force * 4: # Allow stronger flee force
                 steer.scale_to_length(self.max_force * 2)
            return steer
        return pygame.Vector2(0,0)

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

        # --- 1. Needs Update ---
        # Store previous state for boredom calculation
        previous_state = self.state
        previous_action = self.action_type

        # Energy decays based on day/night
        current_energy_decay = ENERGY_DECAY_RATE_NIGHT if self.environment.is_night else ENERGY_DECAY_RATE_DAY
        self.energy -= current_energy_decay * dt
        # Hunger increases
        self.hunger += HUNGER_INCREASE_RATE * dt
        # Thirst increases
        self.thirst += THIRST_INCREASE_RATE * dt
        # Sleepiness increases if awake, decreases if sleeping state
        if self.state != STATE_SLEEPING:
            self.sleepiness += SLEEPINESS_INCREASE_RATE_AWAKE * dt
        else:
            # Decrease sleepiness while in sleeping state
            self.sleepiness -= SLEEPINESS_DECREASE_RATE_ASLEEP * dt
            self.energy += SLEEP_ENERGY_REGEN_RATE * dt # Also slowly regain energy while sleeping

        # Social need decays
        self.social -= SOCIAL_DECAY_RATE * dt
        # Boredom increases if working on same task or idle
        if self.state == STATE_WORKING and self.action_type not in [ACTION_SOCIALIZING, ACTION_LEISURE]: # Assume work is boring
            self.boredom += BOREDOM_INCREASE_RATE_WORKING * dt
        elif self.state == STATE_IDLE:
            self.boredom += BOREDOM_INCREASE_RATE_IDLE * dt
        # Boredom decreases during leisure activity (handled in WORKING state)

        # Clamp needs
        self.energy = max(0, min(MAX_ENERGY, self.energy))
        self.hunger = max(0, min(MAX_HUNGER, self.hunger))
        self.thirst = max(0, min(MAX_THIRST, self.thirst))
        self.sleepiness = max(0, min(MAX_SLEEPINESS, self.sleepiness))
        self.social = max(0, min(MAX_SOCIAL, self.social))
        self.boredom = max(0, min(MAX_BOREDOM, self.boredom))

        # --- 2. Fear Update & Check ---
        # Check for nearby threats
        self._current_threat = self._find_nearest_threat(THREAT_RADIUS)
        if self._current_threat:
            dist_sq = self.pos.distance_squared_to(self._current_threat.pos)
            # Increase fear rapidly when close
            fear_increase = FEAR_INCREASE_PER_SECOND_NEAR_THREAT * (1.0 - (math.sqrt(dist_sq) / THREAT_RADIUS)) * dt
            self.fear = min(MAX_FEAR, self.fear + fear_increase)
        else:
            # Decay fear if no threat nearby
            self.fear = max(0, self.fear - FEAR_DECAY_RATE * dt)

        # Check for death (can happen anytime)
        if self.energy <= 0:
            self.environment.log_event(f"Agent {self.id} ({self.role}) died of exhaustion.")
            self.environment.remove_agent(self)
            return
        # Add death from thirst/hunger if desired
        # if self.thirst >= MAX_THIRST: ... remove agent ...
        # if self.hunger >= MAX_HUNGER: ... remove agent ...

        # --- 3. High Priority Overrides (Fear, Critical Needs) ---
        needs_override = False
        current_max_speed = self.max_speed # Store base speed

        # HIGHEST Priority: FEAR
        if self.fear >= HIGH_FEAR_THRESHOLD and self._current_threat:
            if self.state != STATE_FLEEING:
                self.environment.log_event(f"Agent {self.id} is fleeing!")
                self.state = STATE_FLEEING
                self._clear_targets_and_path() # Stop current task immediately
            needs_override = True
            # Increase speed while fleeing
            current_max_speed = self.max_speed * FLEE_SPEED_MULTIPLIER
        elif self.state == STATE_FLEEING and self.fear < HIGH_FEAR_THRESHOLD * 0.5: # Stop fleeing when fear subsides
             self.environment.log_event(f"Agent {self.id} calmed down.")
             self.state = STATE_IDLE # Go back to idle to decide next action
             self._current_threat = None
             # needs_override remains False here, allowing normal logic to resume next frame

        # Next Priorities: Critical Thirst, Sleepiness, Hunger, Energy
        # Check only if not already fleeing
        if not needs_override:
            # Check Thirst
            if self.thirst >= HIGH_THIRST_THRESHOLD and self.state not in [STATE_MOVING_TO_WATER, STATE_WORKING]:
                target_water = self._find_nearest_water_source()
                if target_water and self._find_path(target_water.grid_pos):
                    self.state = STATE_MOVING_TO_WATER
                    self.target_water = target_water
                    self._clear_targets_and_path(keep_destination=True, keep_water=True) # Keep water target
                    needs_override = True
            # Check Sleepiness
            elif self.sleepiness >= HIGH_SLEEPINESS_THRESHOLD and self.state not in [STATE_MOVING_TO_BASE, STATE_SLEEPING, STATE_WORKING]:
                 if self.environment.base and self._find_path(self.environment.base.grid_pos): # Go to base to sleep
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_SLEEPING_PREP # Set intent
                     self._clear_targets_and_path(keep_destination=True)
                     needs_override = True
            # Check Hunger (Use existing logic)
            elif self.hunger >= HIGH_HUNGER_THRESHOLD and self.state not in [STATE_MOVING_TO_BASE, STATE_WORKING]:
                 if self.environment.base and self._find_path(self.environment.base.grid_pos):
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_EATING
                     self._clear_targets_and_path(keep_destination=True)
                     needs_override = True
            # Check Energy (Use existing logic)
            elif self.energy <= LOW_ENERGY_THRESHOLD and self.state not in [STATE_MOVING_TO_BASE, STATE_WORKING, STATE_SLEEPING]:
                 if self.environment.base and self._find_path(self.environment.base.grid_pos):
                     self.state = STATE_MOVING_TO_BASE
                     self.action_type = ACTION_RESTING
                     self._clear_targets_and_path(keep_destination=True)
                     needs_override = True

        # --- 4. Steering Force Calculation ---
        seek_force = pygame.Vector2(0, 0)
        flee_force = pygame.Vector2(0, 0)
        separation_force = self.separation(nearby_agents) * 1.5

        # Apply flee force if fleeing
        if self.state == STATE_FLEEING and self._current_threat:
            flee_force = self.flee(self._current_threat.pos)
            # Fleeing overrides seeking
        else:
            # Path Following Logic
            path_completed_this_frame = False
            if self.current_path:
                if self.target_pos_world:
                     dist_to_node_sq = self.pos.distance_squared_to(self.target_pos_world)
                     arrival_threshold_sq = (GRID_CELL_SIZE * 0.5)**2
                     if dist_to_node_sq < arrival_threshold_sq:
                         self.path_index += 1
                         if self.path_index < len(self.current_path):
                             next_node_grid = self.current_path[self.path_index]
                             self.target_pos_world = pygame.Vector2(grid_to_world_center(next_node_grid[0], next_node_grid[1]))
                         else: # Reached end of path
                             self.current_path = []
                             self.target_pos_world = None
                             path_completed_this_frame = True
                             if self.destination_grid_pos: self.pos = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))
                             self.velocity *= 0.1 # Dampen on arrival
                else: self.current_path = []; self.path_index = 0 # Clear path if target invalid

            # Determine seek target
            seek_target = self.target_pos_world
            # If path just ended, maybe seek final destination precisely? Depends on action.
            if path_completed_this_frame and self.destination_grid_pos and self.state not in [STATE_WORKING, STATE_SLEEPING]:
                 seek_target = pygame.Vector2(grid_to_world_center(*self.destination_grid_pos))

            # Apply seek force towards the target
            if seek_target:
                 seek_force = self.seek(seek_target)


        # --- 5. State Machine Logic (if no needs override and not fleeing) ---
        if not needs_override and self.state != STATE_FLEEING:
            # --- IDLE State ---
            if self.state == STATE_IDLE:
                # Standard Priority: Return resources if carrying any
                if self.carrying_resource > 0:
                    if self.environment.base and self._find_path(self.environment.base.grid_pos):
                         self.state = STATE_MOVING_TO_BASE; self.action_type = ACTION_RETURNING
                # Role-Based Tasks (Collector/Builder logic from previous step)
                elif self.role == ROLE_BUILDER:
                    build_possible = self.carrying_resource >= BUILD_COST and self.energy > LOW_ENERGY_THRESHOLD + 15
                    if build_possible:
                        build_spot = self._find_build_spot()
                        if build_spot and self._find_path(build_spot): self.state = STATE_MOVING_TO_BUILD
                        elif self.carrying_resource < self.capacity: self._try_find_resource(radius=200) # Collect if can't build
                    elif self.carrying_resource < self.capacity: self._try_find_resource() # Collect if can't build
                elif self.role == ROLE_COLLECTOR:
                    if self.carrying_resource < self.capacity: self._try_find_resource()
                # Default: Collect
                else:
                    if self.carrying_resource < self.capacity: self._try_find_resource()

                # --- Lower Priority Needs (Social, Boredom) - Check if still IDLE ---
                if self.state == STATE_IDLE:
                    if self.social <= LOW_SOCIAL_THRESHOLD:
                        # Try to socialize - move towards base (simple social hub)
                        if self.environment.base and self._find_path(self.environment.base.grid_pos):
                            self.state = STATE_MOVING_TO_SOCIAL_SPOT
                            self.action_type = ACTION_SOCIALIZING # Set intent
                            self.environment.log_event(f"Agent {self.id} lonely, seeking friends.")
                    elif self.boredom >= HIGH_BOREDOM_THRESHOLD:
                        # Seek leisure - wander near base for now
                        if self.environment.base and self._find_path(self.environment.base.grid_pos):
                            self.state = STATE_SEEKING_LEISURE
                            self.action_type = ACTION_LEISURE # Set intent
                            self.environment.log_event(f"Agent {self.id} bored, seeking leisure.")


            # --- Moving States ---
            elif self.state == STATE_MOVING_TO_RESOURCE and path_completed_this_frame:
                if self.target_resource and self.target_resource.quantity > 0 and self.environment.is_resource_targeted_by(self.target_resource, self):
                    self.state = STATE_WORKING; self.action_type = ACTION_COLLECTING
                    self.action_timer = 1.0 + random.uniform(-0.2, 0.2)
                else: self._clear_targets_and_path(); self.state = STATE_IDLE # Go idle if failed

            elif self.state == STATE_MOVING_TO_BUILD and path_completed_this_frame:
                current_grid_pos=world_to_grid(*self.pos); build_target=self.destination_grid_pos
                if build_target and current_grid_pos == build_target and self.environment.is_buildable(*build_target):
                    self.state = STATE_WORKING; self.action_type = ACTION_BUILDING; self.action_timer = BUILD_TIME
                else: self._clear_targets_and_path(); self.state = STATE_IDLE

            elif self.state == STATE_MOVING_TO_WATER and path_completed_this_frame:
                # Check if arrived near water source
                if self.target_water and self.pos.distance_to(self.target_water.pos) < GRID_CELL_SIZE * 1.5:
                    self.state = STATE_WORKING
                    self.action_type = ACTION_DRINKING
                    self.action_timer = DRINK_TIME
                else: self._clear_targets_and_path(); self.state = STATE_IDLE # Failed to reach water?

            elif self.state == STATE_MOVING_TO_SOCIAL_SPOT and path_completed_this_frame:
                # Arrived at base (social hub), start socializing
                self.state = STATE_WORKING
                self.action_type = ACTION_SOCIALIZING
                self.action_timer = SOCIALIZE_TIME
                # Find a nearby agent target? Or just be near base? Keep simple: be near base.

            elif self.state == STATE_SEEKING_LEISURE and path_completed_this_frame:
                 # Arrived at base (leisure hub), start leisure action
                self.state = STATE_WORKING
                self.action_type = ACTION_LEISURE
                self.action_timer = LEISURE_TIME

            elif self.state == STATE_MOVING_TO_BASE and path_completed_this_frame:
                 # Handle different reasons for going to base
                 if self.action_type == ACTION_RETURNING:
                     self.environment.add_base_resources(self.carrying_resource)
                     self.environment.log_event(f"Agent {self.id} ({self.role}) deposited {self.carrying_resource} res.")
                     self.carrying_resource = 0; self.action_type = None; self.state = STATE_IDLE
                 elif self.action_type == ACTION_EATING: self.state = STATE_WORKING; self.action_timer = EAT_TIME
                 elif self.action_type == ACTION_RESTING: self.state = STATE_WORKING; self.action_timer = REST_TIME
                 elif self.action_type == ACTION_SLEEPING_PREP:
                      self.state = STATE_SLEEPING # Transition to sleeping state
                      # No timer needed for sleeping state itself, maybe duration based on need?
                      # For now, sleep until sleepiness is low or disturbed.
                 else: self.action_type = None; self.state = STATE_IDLE # Unknown reason

            # --- WORKING State (Performing Timed Actions) ---
            elif self.state == STATE_WORKING:
                self.action_timer -= dt
                action_finished_naturally = self.action_timer <= 0
                action_interrupted = False # Flag if needs/fear interrupt

                # Interrupt checks (Fear has highest priority)
                if self.fear >= HIGH_FEAR_THRESHOLD:
                    action_interrupted = True
                    self.environment.log_event(f"Agent {self.id} work interrupted by fear!")
                # Add checks for other critical needs if they should interrupt work (e.g., extreme thirst)
                # elif self.thirst >= MAX_THIRST * 0.98: action_interrupted = True ...

                if action_interrupted:
                    # State change (e.g., to FLEEING) is handled by the override logic earlier
                    # Just need to clean up the current action here
                    self._clear_targets_and_path() # Stop current path/targets
                    # State will change automatically due to override check next frame
                elif action_finished_naturally:
                    action_done = False # Track if action completed its effect
                    if self.action_type == ACTION_COLLECTING:
                        if self.target_resource and self.target_resource.quantity > 0:
                            collected = self.target_resource.collect(1)
                            if collected > 0:
                                self.carrying_resource += collected
                                if self.carrying_resource >= self.capacity or self.target_resource.quantity == 0:
                                    if self.target_resource.quantity == 0: self.environment.log_event(f"Res {self.target_resource.id} depleted.")
                                    self._clear_targets_and_path(); action_done = True
                                else: self.action_timer = 1.0 + random.uniform(-0.2, 0.2) # Collect more
                            else: self._clear_targets_and_path(); action_done = True # Collect failed
                        else: self._clear_targets_and_path(); action_done = True # Res gone
                    elif self.action_type == ACTION_BUILDING:
                         build_pos = self.destination_grid_pos
                         if build_pos and self.carrying_resource >= BUILD_COST:
                              success = self.environment.build_wall(*build_pos, self)
                              if success: self.carrying_resource -= BUILD_COST; self.environment.log_event(f"Agent {self.id} ({self.role}) built wall @ {build_pos}.")
                         self._clear_targets_and_path(); action_done = True
                    elif self.action_type == ACTION_EATING: self.hunger = max(0, self.hunger - EAT_AMOUNT); action_done = True
                    elif self.action_type == ACTION_RESTING: self.energy = min(MAX_ENERGY, self.energy + REST_AMOUNT); action_done = True
                    elif self.action_type == ACTION_DRINKING: self.thirst = max(0, self.thirst - DRINK_AMOUNT); action_done = True
                    elif self.action_type == ACTION_SOCIALIZING:
                        # Simple: just increases social need
                        self.social = min(MAX_SOCIAL, self.social + SOCIAL_GAIN_RATE * SOCIALIZE_TIME) # Gain based on duration
                        # Could add check for nearby agents here for more realistic gain
                        action_done = True
                    elif self.action_type == ACTION_LEISURE:
                        self.boredom = max(0, self.boredom - BOREDOM_DECREASE_WHILE_LEISURE * LEISURE_TIME) # Decrease based on duration
                        action_done = True
                    else: action_done = True # Unknown action

                    if action_done:
                        self.action_type = None
                        self.state = STATE_IDLE # Go idle after completing work
                # Else: action timer still running, continue working

            # --- SLEEPING State ---
            elif self.state == STATE_SLEEPING:
                # Remain sleeping until sleepiness is low or fear is high
                if self.fear >= HIGH_FEAR_THRESHOLD * 0.8: # Wake up if moderately scared
                    self.environment.log_event(f"Agent {self.id} woke up scared!")
                    self.state = STATE_IDLE # Will likely transition to FLEEING immediately
                elif self.sleepiness <= 5: # Wake up threshold
                    self.environment.log_event(f"Agent {self.id} finished sleeping.")
                    self.state = STATE_IDLE
                # While sleeping, velocity should be near zero (handled below)


        # --- 6. Apply Forces & Update Movement ---
        total_force = pygame.Vector2(0, 0)
        if self.state == STATE_FLEEING:
            total_force = flee_force + separation_force # Fleeing ignores seeking
        elif self.state != STATE_SLEEPING and self.state != STATE_WORKING: # Don't seek/separate much when sleeping/working
            total_force = seek_force + separation_force

        # Apply acceleration (F/m * dt, assume m=1)
        self.velocity += total_force * dt

        # Limit velocity based on current max speed (potentially boosted by fleeing)
        vel_mag_sq = self.velocity.length_squared()
        max_speed_sq = (current_max_speed * current_max_speed)
        if vel_mag_sq > max_speed_sq:
            self.velocity.scale_to_length(current_max_speed)

        # Apply damping based on state
        if self.state == STATE_WORKING or self.state == STATE_SLEEPING:
             self.velocity *= 0.05 # Slow down significantly
        elif self.state == STATE_IDLE and not self.target_pos_world and not self.current_path:
             self.velocity *= 0.85 # Slow down if idle with no target

        # Update position
        # Don't move much if sleeping
        if self.state != STATE_SLEEPING:
            self.pos += self.velocity * dt

        # --- 7. Boundary Constraints ---
        self.pos.x = max(AGENT_SIZE, min(SIM_WIDTH - AGENT_SIZE, self.pos.x))
        self.pos.y = max(AGENT_SIZE, min(SIM_HEIGHT - AGENT_SIZE, self.pos.y))

        # Update Rect for Quadtree and drawing
        self.update_rect()

    # --- Helper Methods ---
    def _clear_targets_and_path(self, keep_destination=False, keep_water=False, keep_resource=False):
        """Clears current path and targets, with options to keep specific ones."""
        self.current_path = []
        self.path_index = 0
        self.target_pos_world = None
        if not keep_destination: self.destination_grid_pos = None
        if not keep_water: self.target_water = None
        if not keep_resource: self._release_resource_target() # Use existing method
        self.target_agent = None # Clear social target

    def _try_find_resource(self, radius=300):
        """Attempts to find and path to the best available resource."""
        resource_target = self._find_best_available_resource(max_search_radius=radius)
        if resource_target and self._find_path(resource_target.grid_pos):
            self.state = STATE_MOVING_TO_RESOURCE
            self.target_resource = resource_target
            self.environment.mark_resource_targeted(resource_target, self)
            return True
        return False

    def _find_nearest_water_source(self, max_radius=500):
        """Finds the closest WaterSource entity."""
        nearby = self.environment.quadtree.query_radius(self.pos, max_radius)
        water_sources = [e for e in nearby if isinstance(e, entities.WaterSource)]
        closest_water = None
        min_dist_sq = float('inf')
        for water in water_sources:
            dist_sq = self.pos.distance_squared_to(water.pos)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_water = water
        return closest_water

    def _find_nearby_agents_for_social(self, radius=SOCIAL_INTERACTION_RANGE):
        """Finds other agents within social range."""
        nearby = self.environment.quadtree.query_radius(self.pos, radius)
        # Filter for other agents, excluding self
        social_partners = [a for a in nearby if isinstance(a, Agent) and a is not self]
        return social_partners

    def _find_nearest_threat(self, radius=THREAT_RADIUS * 1.2): # Search slightly larger radius
         """Finds the closest Threat entity."""
         # Use environment's threat list or quadtree query if threats are in it
         threats = self.environment.get_threats_near(self.pos, radius) # Assumes Environment method exists
         closest_threat = None
         min_dist_sq = radius * radius
         for threat in threats:
             dist_sq = self.pos.distance_squared_to(threat.pos)
             if dist_sq < min_dist_sq:
                 min_dist_sq = dist_sq
                 closest_threat = threat
         return closest_threat

    def _release_resource_target(self):
        if self.target_resource:
            self.environment.mark_resource_available(self.target_resource)
            self.target_resource = None

    def _find_best_available_resource(self, max_search_radius=300):
        # ... (same as before) ...
        best_score = -float('inf'); best_resource = None
        potential_resources = self.environment.quadtree.query_radius(self.pos, max_search_radius)
        available_resources = [res for res in potential_resources if isinstance(res, entities.Resource) and res.quantity > 0 and res.id not in self.environment.targeted_resources]
        if not available_resources: return None
        weight_quantity = 1.5; weight_distance = -0.05
        for resource in available_resources:
            dist_sq = self.pos.distance_squared_to(resource.pos); dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1
            score = (weight_quantity * resource.quantity) + (weight_distance * dist)
            if score > best_score: best_score = score; best_resource = resource
        return best_resource

    def _find_build_spot(self):
        # ... (same as before) ...
        my_grid_pos = world_to_grid(self.pos.x, self.pos.y)
        if my_grid_pos is None: return None
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]; random.shuffle(neighbors)
        for dx, dy in neighbors:
            check_x, check_y = my_grid_pos[0] + dx, my_grid_pos[1] + dy
            if self.environment.is_buildable(check_x, check_y): return (check_x, check_y)
        return None

    def draw(self, screen):
         """Draws the agent on the screen."""
         if self.pos.x >= SIM_WIDTH or self.pos.y >= SIM_HEIGHT or self.pos.x < 0 or self.pos.y < 0: return

         center_tuple = (int(self.pos.x), int(self.pos.y))
         pygame.draw.circle(screen, self.color, center_tuple, AGENT_SIZE) # Color based on role

         # --- State Indicator (Inner Circle) ---
         state_color = WHITE # Default for IDLE
         if self.state == STATE_FLEEING: state_color = RED # Override color when fleeing
         elif self.state == STATE_SLEEPING: state_color = PURPLE # Distinct color for sleeping
         # Other state colors (adjust as needed)
         elif self.state == STATE_MOVING_TO_RESOURCE: state_color = CYAN
         elif self.state == STATE_MOVING_TO_BUILD: state_color = GREY # Use grey for build move? Builder is Orange.
         elif self.state == STATE_MOVING_TO_WATER: state_color = BLUE # Moving to water
         elif self.state == STATE_MOVING_TO_SOCIAL_SPOT: state_color = YELLOW
         elif self.state == STATE_SEEKING_LEISURE: state_color = GREEN
         elif self.state == STATE_MOVING_TO_BASE:
             if self.action_type == ACTION_RETURNING: state_color = LIGHT_GREY
             elif self.action_type == ACTION_EATING: state_color = DARK_GREY # Use different greys?
             elif self.action_type == ACTION_RESTING: state_color = DARK_GREY
             elif self.action_type == ACTION_SLEEPING_PREP: state_color = DARK_GREY
             else: state_color = DARK_GREY
         elif self.state == STATE_WORKING:
             if self.action_type == ACTION_COLLECTING: state_color = CYAN
             elif self.action_type == ACTION_BUILDING: state_color = GREY
             elif self.action_type == ACTION_EATING: state_color = DARK_GREY
             elif self.action_type == ACTION_RESTING: state_color = DARK_GREY
             elif self.action_type == ACTION_DRINKING: state_color = BLUE
             elif self.action_type == ACTION_SOCIALIZING: state_color = YELLOW
             elif self.action_type == ACTION_LEISURE: state_color = GREEN
             else: state_color = DARK_GREY
         pygame.draw.circle(screen, state_color, center_tuple, max(1, AGENT_SIZE // 2))

         # --- Draw Needs Bars --- (Thirst, Sleepiness added)
         bar_width = AGENT_SIZE * 2; bar_height = 2; bar_padding = 1
         bar_x = self.pos.x - bar_width / 2
         bar_y_start = self.pos.y + AGENT_SIZE + 2

         # Energy Bar (Green)
         y_pos = bar_y_start
         energy_ratio = self.energy / MAX_ENERGY
         pygame.draw.rect(screen, DARK_GREY, (bar_x, y_pos, bar_width, bar_height))
         pygame.draw.rect(screen, GREEN, (bar_x, y_pos, bar_width * energy_ratio, bar_height))

         # Hunger Bar (Red)
         y_pos += bar_height + bar_padding
         hunger_ratio = self.hunger / MAX_HUNGER
         pygame.draw.rect(screen, DARK_GREY, (bar_x, y_pos, bar_width, bar_height))
         pygame.draw.rect(screen, RED, (bar_x, y_pos, bar_width * hunger_ratio, bar_height))

         # Thirst Bar (Blue)
         y_pos += bar_height + bar_padding
         thirst_ratio = self.thirst / MAX_THIRST
         pygame.draw.rect(screen, DARK_GREY, (bar_x, y_pos, bar_width, bar_height))
         pygame.draw.rect(screen, BLUE, (bar_x, y_pos, bar_width * thirst_ratio, bar_height))

         # Sleepiness Bar (Purple)
         y_pos += bar_height + bar_padding
         sleep_ratio = self.sleepiness / MAX_SLEEPINESS
         pygame.draw.rect(screen, DARK_GREY, (bar_x, y_pos, bar_width, bar_height))
         pygame.draw.rect(screen, PURPLE, (bar_x, y_pos, bar_width * sleep_ratio, bar_height))