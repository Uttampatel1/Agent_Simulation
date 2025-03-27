import pygame
import random
import math
import time

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
AGENT_SIZE = 10
AGENT_SPEED = 50  # Pixels per second
RESOURCE_SIZE = 15
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Agent States
STATE_IDLE = "IDLE"
STATE_MOVING = "MOVING"
STATE_WORKING = "WORKING" # Simple example: just pausing

# --- Helper Functions ---
def distance(pos1, pos2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# --- Classes ---

class Resource:
    """Represents a resource or point of interest in the environment."""
    _id_counter = 0
    def __init__(self, x, y):
        self.id = Resource._id_counter
        Resource._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.color = GREEN

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos.x - RESOURCE_SIZE // 2, self.pos.y - RESOURCE_SIZE // 2, RESOURCE_SIZE, RESOURCE_SIZE))

class Agent:
    """Represents an agent in the simulation."""
    _id_counter = 0
    def __init__(self, x, y, environment):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.pos = pygame.Vector2(x, y)
        self.environment = environment # Reference to the environment
        self.color = BLUE
        self.state = STATE_IDLE
        self.target_pos = None
        self.work_timer = 0
        self.target_resource = None # Keep track of which resource is targeted

    def set_target(self, target_pos, target_resource=None):
        """Sets a new movement target."""
        if target_pos:
            self.target_pos = pygame.Vector2(target_pos)
            self.state = STATE_MOVING
            self.target_resource = target_resource
            # print(f"Agent {self.id}: New target {self.target_pos}")
        else:
            self.target_pos = None
            self.state = STATE_IDLE
            self.target_resource = None

    def update(self, dt):
        """Updates the agent's state and position."""
        # --- State Machine ---
        if self.state == STATE_IDLE:
            # Find something to do - e.g., find the nearest available resource
            nearest_resource = self.find_nearest_available_resource()
            if nearest_resource:
                self.set_target(nearest_resource.pos, nearest_resource)
                self.environment.mark_resource_targeted(nearest_resource, self) # Mark resource as targeted
            else:
                 # Or maybe just move randomly if no resources?
                 # random_target = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
                 # self.set_target(random_target)
                 pass # Stay idle if nothing found


        elif self.state == STATE_MOVING:
            if self.target_pos:
                direction = self.target_pos - self.pos
                dist = direction.length()

                if dist < 5: # Close enough to target
                    self.pos = self.target_pos # Snap to target
                    if self.target_resource: # Arrived at a resource?
                         self.state = STATE_WORKING
                         self.work_timer = random.uniform(1.0, 3.0) # Work for 1-3 seconds
                         print(f"Agent {self.id}: Arrived at resource {self.target_resource.id}. Working for {self.work_timer:.2f}s")
                    else: # Arrived at a random point
                         self.state = STATE_IDLE
                    self.target_pos = None
                    # self.target_resource remains until work is done

                else:
                    # Move towards target
                    direction.normalize_ip() # Normalize in-place
                    move_vector = direction * AGENT_SPEED * dt
                    # Check if move exceeds distance, if so, clamp it
                    if move_vector.length() > dist:
                        self.pos = self.target_pos
                    else:
                        self.pos += move_vector
            else:
                # Should not happen in MOVING state, reset to IDLE
                self.state = STATE_IDLE

        elif self.state == STATE_WORKING:
            self.work_timer -= dt
            if self.work_timer <= 0:
                print(f"Agent {self.id}: Finished working at resource {self.target_resource.id}")
                if self.target_resource:
                    self.environment.mark_resource_available(self.target_resource) # Make resource available again
                self.target_resource = None
                self.state = STATE_IDLE # Go back to idle to find new task

        # --- Boundary Check ---
        self.pos.x = max(0 + AGENT_SIZE//2, min(SCREEN_WIDTH - AGENT_SIZE//2, self.pos.x))
        self.pos.y = max(0 + AGENT_SIZE//2, min(SCREEN_HEIGHT - AGENT_SIZE//2, self.pos.y))


    def find_nearest_available_resource(self):
        """Finds the closest resource that isn't currently targeted."""
        min_dist = float('inf')
        nearest = None
        for resource in self.environment.get_available_resources():
            d = distance(self.pos, resource.pos)
            if d < min_dist:
                min_dist = d
                nearest = resource
        return nearest

    def draw(self, screen):
        """Draws the agent on the screen."""
        # Draw body
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), AGENT_SIZE // 2)
        # Draw state indicator (optional)
        state_color = WHITE
        if self.state == STATE_MOVING: state_color = YELLOW
        elif self.state == STATE_WORKING: state_color = RED
        pygame.draw.circle(screen, state_color, (int(self.pos.x), int(self.pos.y)), AGENT_SIZE // 4)
        # Draw target line (optional)
        # if self.state == STATE_MOVING and self.target_pos:
        #    pygame.draw.line(screen, RED, self.pos, self.target_pos, 1)


class Environment:
    """Manages the simulation space, agents, and resources."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.resources = []
        self.targeted_resources = {} # resource_id -> agent targeting it

    def add_agent(self, agent):
        """Adds an agent to the environment."""
        self.agents.append(agent)

    def add_resource(self, resource):
        """Adds a resource to the environment."""
        self.resources.append(resource)

    def create_random_resource(self):
        """Creates a resource at a random position."""
        x = random.randint(RESOURCE_SIZE, self.width - RESOURCE_SIZE)
        y = random.randint(RESOURCE_SIZE, self.height - RESOURCE_SIZE)
        self.add_resource(Resource(x,y))

    def create_random_agent(self):
        """Creates an agent at a random position."""
        x = random.randint(AGENT_SIZE, self.width - AGENT_SIZE)
        y = random.randint(AGENT_SIZE, self.height - AGENT_SIZE)
        self.add_agent(Agent(x, y, self)) # Pass environment reference

    def mark_resource_targeted(self, resource, agent):
        """Marks a resource as being targeted by an agent."""
        if resource and resource.id not in self.targeted_resources:
            self.targeted_resources[resource.id] = agent.id
            resource.color = YELLOW # Visually show it's targeted
            # print(f"Resource {resource.id} targeted by Agent {agent.id}")

    def mark_resource_available(self, resource):
        """Marks a resource as available again."""
        if resource and resource.id in self.targeted_resources:
            del self.targeted_resources[resource.id]
            resource.color = GREEN # Visually show it's available
            # print(f"Resource {resource.id} is now available")

    def get_available_resources(self):
        """Returns a list of resources not currently targeted."""
        return [res for res in self.resources if res.id not in self.targeted_resources]

    def update(self, dt):
        """Updates all agents in the environment."""
        for agent in self.agents:
            agent.update(dt)

    def draw(self, screen):
        """Draws the environment (background, resources, agents)."""
        # Draw background (optional, can be done in main loop)
        # screen.fill(BLACK)

        # Draw resources
        for resource in self.resources:
            resource.draw(screen)

        # Draw agents
        for agent in self.agents:
            agent.draw(screen)


# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Real-Time Agent Simulation")
    clock = pygame.time.Clock()

    # --- Create Environment and Entities ---
    environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Create initial resources
    for _ in range(5):
        environment.create_random_resource()

    # Create initial agents
    for _ in range(10):
        environment.create_random_agent()

    running = True
    while running:
        # --- Delta Time Calculation ---
        # dt = amount of time passed since last frame in seconds
        dt = clock.tick(FPS) / 1000.0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_a: # Add agent
                    environment.create_random_agent()
                    print(f"Added Agent. Total: {len(environment.agents)}")
                if event.key == pygame.K_r: # Add resource
                     environment.create_random_resource()
                     print(f"Added Resource. Total: {len(environment.resources)}")


        # --- Updates ---
        environment.update(dt)

        # --- Drawing ---
        screen.fill(BLACK)         # Clear screen
        environment.draw(screen)   # Draw environment contents
        pygame.display.flip()      # Update the display

    pygame.quit()

if __name__ == '__main__':
    main()