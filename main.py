import pygame
import sys
import random

# Import project modules
from constants import *
from utils import *
from environment import Environment
from ui_manager import UIManager
from persistence import save_simulation, load_simulation
from entities import Base # For initial setup

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
    if not environment.base: print("Critical Error: No base."); pygame.quit(); sys.exit()
    # Clear around base (same as before)
    if environment.base.grid_pos:
        gx, gy = environment.base.grid_pos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx, cy = gx+dx, gy+dy
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT and environment.grid[cy][cx] == OBSTACLE_COST:
                     environment.grid[cy][cx] = TERRAIN_PLAINS
                     environment.obstacles = [o for o in environment.obstacles if o.grid_pos != (cx, cy)]

    # Add initial features (maybe add water?)
    environment.create_obstacle_line(SIM_WIDTH*0.6, SIM_HEIGHT*0.1, SIM_WIDTH*0.7, SIM_HEIGHT*0.9)
    environment.create_obstacle_line(SIM_WIDTH*0.1, SIM_HEIGHT*0.8, SIM_WIDTH*0.5, SIM_HEIGHT*0.7)
    # Add a water source or two
    environment.create_random_entity('water')
    environment.create_random_entity('water')
    # Add some threats?
    environment.create_random_entity('threat')
    # Add resources/agents
    for _ in range(40): environment.create_random_entity('resource')
    for _ in range(60): environment.create_random_entity('agent')
    environment.log_event("Simulation Started.")

    # --- Main Loop ---
    running = True
    while running:
        base_dt = clock.tick(FPS) / 1000.0; dt = min(base_dt, 0.1)
        mouse_pos = pygame.mouse.get_pos()
        sim_area_rect = pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT); mouse_in_sim = sim_area_rect.collidepoint(mouse_pos)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            # --- Keyboard (same as before) ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_p: game_state['paused'] = not game_state['paused']
                elif event.key == pygame.K_F5: save_simulation(environment)
                elif event.key == pygame.K_F9: # Load
                    loaded_env, selected_id = load_simulation()
                    if loaded_env:
                        environment = loaded_env; ui_manager = UIManager(environment, fonts) # Recreate UI
                        environment.selected_agent = None # Find selected agent
                        if selected_id is not None:
                             for agent in environment.agents:
                                 if agent.id == selected_id: environment.selected_agent = agent; break
                        environment.log_event("Simulation Loaded.")
                elif event.key == pygame.K_g: game_state['draw_grid'] = not game_state['draw_grid']
                elif event.key == pygame.K_q: game_state['draw_quadtree'] = not game_state['draw_quadtree']
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    environment.brush_size = clamp(environment.brush_size - 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE); ui_manager._update_button_selected_state()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    environment.brush_size = clamp(environment.brush_size + 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE); ui_manager._update_button_selected_state()

            # --- Mouse Handling (Added Spawn Water/Threat) ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    if mouse_pos[0] >= UI_PANEL_X: # UI Click
                        action_result = ui_manager.handle_click(mouse_pos, game_state)
                        if action_result == "save": save_simulation(environment)
                        elif action_result == "load": print("Load requested via button. Press F9.")
                    elif mouse_in_sim: # Simulation Area Click
                        current_brush = environment.paint_brush
                        if current_brush == BRUSH_SELECT: environment.select_agent_at(mouse_pos)
                        elif current_brush in PAINT_BRUSHES: game_state['is_painting'] = True; environment.paint_terrain(mouse_pos)
                        elif current_brush == BRUSH_SPAWN_AGENT: environment.create_agent_at(mouse_pos[0], mouse_pos[1])
                        elif current_brush == BRUSH_SPAWN_RESOURCE: environment.create_resource_at(mouse_pos[0], mouse_pos[1])
                        elif current_brush == BRUSH_SPAWN_WATER: environment.create_water_source_at(mouse_pos[0], mouse_pos[1]) # New
                        elif current_brush == BRUSH_SPAWN_THREAT: environment.create_threat_at(mouse_pos[0], mouse_pos[1])    # New
                        elif current_brush == BRUSH_DELETE:
                             grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
                             if grid_pos: environment.delete_entity_at(grid_pos[0], grid_pos[1]); game_state['is_painting'] = True

            elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1: game_state['is_painting'] = False

            elif event.type == pygame.MOUSEMOTION:
                 if game_state['is_painting'] and mouse_in_sim:
                      current_brush = environment.paint_brush
                      if current_brush in PAINT_BRUSHES: environment.paint_terrain(mouse_pos)
                      elif current_brush == BRUSH_DELETE:
                           grid_pos = world_to_grid(mouse_pos[0], mouse_pos[1])
                           if grid_pos: environment.delete_entity_at(grid_pos[0], grid_pos[1])

        # --- Updates ---
        if not game_state['paused']:
            environment.update(dt) # Pass base dt

        # --- Drawing ---
        screen.fill(BLACK)
        environment.draw_sim_area(screen, game_state['draw_grid'], game_state['draw_quadtree'])
        ui_manager.draw(screen, clock, game_state)
        pygame.display.flip()

    # --- Cleanup ---
    pygame.quit(); sys.exit()

if __name__ == '__main__':
    main()