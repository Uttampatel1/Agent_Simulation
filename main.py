import pygame
import sys
import random

# Import project modules
from constants import *
from utils import * # world_to_grid etc.
from environment import Environment
from ui_manager import UIManager
from persistence import save_simulation, load_simulation
# Import entity classes only if needed for direct checks/creation here
# from agent import Agent
from entities import Base #, Resource

def main():
    # --- Pygame & Font Initialization ---
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enhanced Agent Simulation v5") # Update caption
    clock = pygame.time.Clock()

    # Load fonts into a dictionary
    try:
        fonts = {
            'small': pygame.font.SysFont(None, UI_FONT_SIZE_SMALL),
            'medium': pygame.font.SysFont(None, UI_FONT_SIZE_MEDIUM),
            'large': pygame.font.SysFont(None, UI_FONT_SIZE_LARGE),
        }
    except Exception as e:
        print(f"Error loading system font: {e}. Using default.")
        fonts = {
            'small': pygame.font.Font(None, UI_FONT_SIZE_SMALL),
            'medium': pygame.font.Font(None, UI_FONT_SIZE_MEDIUM),
            'large': pygame.font.Font(None, UI_FONT_SIZE_LARGE),
        }

    # --- Game State Dictionary ---
    game_state = {
        'paused': False,
        'draw_grid': True,
        'draw_quadtree': False,
        'is_painting': False, # For mouse dragging (terrain paint, delete)
    }

    # --- Create Environment & UI ---
    # Initialize environment first
    environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT, SIM_WIDTH, SIM_HEIGHT)
    # Then initialize UI Manager, passing environment reference and fonts
    ui_manager = UIManager(environment, fonts)

    # --- Initial Population ---
    base_x, base_y = SIM_WIDTH // 4, SIM_HEIGHT // 2
    environment.set_base(Base(base_x, base_y))
    if not environment.base:
        print("Critical Error: Failed to place base. Exiting.")
        pygame.quit()
        sys.exit()

    # Ensure area around base is clear
    if environment.base.grid_pos:
         gx, gy = environment.base.grid_pos
         for dx in range(-1, 2):
             for dy in range(-1, 2):
                  clear_x, clear_y = gx + dx, gy + dy
                  if 0 <= clear_x < GRID_WIDTH and 0 <= clear_y < GRID_HEIGHT:
                      # Only clear if it's currently an obstacle
                      if environment.grid[clear_y][clear_x] == OBSTACLE_COST:
                           environment.grid[clear_y][clear_x] = TERRAIN_PLAINS
                           # Also remove from obstacle list if present
                           environment.obstacles = [o for o in environment.obstacles if o.grid_pos != (clear_x, clear_y)]

    # Add some initial features
    environment.create_obstacle_line(SIM_WIDTH * 0.6, SIM_HEIGHT * 0.1, SIM_WIDTH * 0.7, SIM_HEIGHT * 0.9)
    environment.create_obstacle_line(SIM_WIDTH * 0.1, SIM_HEIGHT * 0.8, SIM_WIDTH * 0.5, SIM_HEIGHT * 0.7)
    for _ in range(40): environment.create_random_entity('resource')
    for _ in range(60): environment.create_random_entity('agent') # Roles assigned in Agent.__init__
    environment.log_event("Simulation Started.")

    # --- Main Loop ---
    running = True
    while running:
        # Calculate delta time
        base_dt = clock.tick(FPS) / 1000.0
        dt = min(base_dt, 0.1) # Clamp dt to prevent large jumps

        # Get mouse position once per frame
        mouse_pos = pygame.mouse.get_pos()
        # Check if mouse is within simulation area
        sim_area_rect = pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT)
        mouse_in_sim = sim_area_rect.collidepoint(mouse_pos)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- Keyboard Input ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_p: game_state['paused'] = not game_state['paused'] # Pause toggle
                elif event.key == pygame.K_F5: save_simulation(environment) # Save
                elif event.key == pygame.K_F9: # Load
                    loaded_env, selected_id = load_simulation()
                    if loaded_env:
                        environment = loaded_env # Replace environment instance
                        # --- Restore Transients after Load ---
                        # Recreate UI Manager with the new environment reference
                        ui_manager = UIManager(environment, fonts)
                        # Find and re-select the agent based on the loaded ID
                        environment.selected_agent = None
                        if selected_id is not None:
                             for agent in environment.agents:
                                 if agent.id == selected_id:
                                     environment.selected_agent = agent
                                     break
                        environment.log_event("Simulation Loaded.")
                        print("Loaded environment assigned and UI/Selection restored.") # Console feedback
                # Debug toggles
                elif event.key == pygame.K_g: game_state['draw_grid'] = not game_state['draw_grid']
                elif event.key == pygame.K_q: game_state['draw_quadtree'] = not game_state['draw_quadtree']
                # Brush size adjustment keys
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    environment.brush_size = clamp(environment.brush_size - 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
                    ui_manager._update_button_selected_state() # Update UI display/tooltips
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS: # Use equals/plus key
                    environment.brush_size = clamp(environment.brush_size + 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
                    ui_manager._update_button_selected_state() # Update UI display/tooltips

            # --- Mouse Input ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Mouse Button Click
                    # Check UI Panel First
                    if mouse_pos[0] >= UI_PANEL_X:
                        action_result = ui_manager.handle_click(mouse_pos, game_state)
                        # Handle signals from UI click
                        if action_result == "save":
                            save_simulation(environment)
                        elif action_result == "load":
                            # Prefer F9 for load consistency, provide feedback
                            print("Load requested via button. Press F9 to load.")
                            environment.log_event("Load requested via button (Use F9).")

                    # Check Simulation Area if not UI click
                    elif mouse_in_sim:
                        current_brush = environment.paint_brush
                        # Handle action based on selected brush
                        if current_brush == BRUSH_SELECT:
                            environment.select_agent_at(mouse_pos)
                        elif current_brush in PAINT_BRUSHES:
                            game_state['is_painting'] = True # Start drag painting
                            environment.paint_terrain(mouse_pos) # Paint single cell on click
                        elif current_brush == BRUSH_SPAWN_AGENT:
                             environment.create_agent_at(mouse_pos[0], mouse_pos[1]) # Spawn on click
                        elif current_brush == BRUSH_SPAWN_RESOURCE:
                             environment.create_resource_at(mouse_pos[0], mouse_pos[1]) # Spawn on click
                        elif current_brush == BRUSH_DELETE:
                             grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
                             if grid_pos:
                                 environment.delete_entity_at(grid_pos[0], grid_pos[1]) # Delete on click
                                 game_state['is_painting'] = True # Enable drag-delete

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1: # Left Mouse Button Release
                      game_state['is_painting'] = False # Stop drag painting/deleting

            elif event.type == pygame.MOUSEMOTION:
                 # Handle dragging actions if painting state is active
                 if game_state['is_painting'] and mouse_in_sim:
                      current_brush = environment.paint_brush
                      if current_brush in PAINT_BRUSHES:
                           environment.paint_terrain(mouse_pos) # Paint while dragging
                      elif current_brush == BRUSH_DELETE:
                           grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
                           if grid_pos:
                               environment.delete_entity_at(grid_pos[0], grid_pos[1]) # Delete while dragging

        # --- Simulation Updates ---
        # Update environment and its contents only if not paused
        # The environment's update method now handles applying the speed multiplier internally
        if not game_state['paused']:
            environment.update(dt)

        # --- Drawing ---
        screen.fill(BLACK) # Clear screen

        # Draw simulation area contents (grid, entities, night overlay)
        environment.draw_sim_area(screen, game_state['draw_grid'], game_state['draw_quadtree'])

        # Draw UI panel contents (minimap, buttons, info, log)
        ui_manager.draw(screen, clock, game_state)

        # Update the full display surface
        pygame.display.flip()

    # --- Cleanup ---
    pygame.quit()
    sys.exit()

# --- Run ---
if __name__ == '__main__':
    main()