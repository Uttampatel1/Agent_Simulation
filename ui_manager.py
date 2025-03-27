import pygame
import math
from collections import deque
from constants import *
from ui_elements import Button
from utils import clamp
import entities # For type checking
from agent import Agent # For type checking

class UIManager:
    """Manages the creation, drawing, and interaction of the UI panel."""
    def __init__(self, environment, fonts):
        self.environment = environment
        self.fonts = fonts
        self.buttons = []
        self.active_brush_tooltip = ""

        # Minimap Setup (same as before)
        self.minimap_surface = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT)).convert()
        self.minimap_rect = pygame.Rect(MINIMAP_X, MINIMAP_Y, MINIMAP_WIDTH, MINIMAP_HEIGHT)
        self.map_scale_x = MINIMAP_WIDTH / SIM_WIDTH
        self.map_scale_y = MINIMAP_HEIGHT / SIM_HEIGHT
        self.map_grid_step_x = max(1, int(GRID_WIDTH / MINIMAP_WIDTH) + 1)
        self.map_grid_step_y = max(1, int(GRID_HEIGHT / MINIMAP_HEIGHT) + 1)

        self._setup_ui_buttons() # Calls _update_button_selected_state inside

    def _setup_ui_buttons(self):
        """Creates the UI buttons, including new Paint Water, Spawn Water/Threat."""
        self.buttons = []
        font = self.fonts['small']
        bw = (UI_PANEL_WIDTH - 3 * BUTTON_PADDING) // 2
        bh = BUTTON_HEIGHT
        bx = UI_PANEL_X + BUTTON_PADDING
        by = MINIMAP_Y + MINIMAP_HEIGHT + BUTTON_PADDING * 2 # Start below minimap

        # --- Sim Control, Debug, Brush Size --- (Same as before)
        self.buttons.append(Button(bx, by, bw, bh, "Pause", "toggle_pause", font, "P / Click")); self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Play", "play", font, "Resumes Sim")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, bw, bh, "Slow (-)", "speed_down", font, "Decrease Sim Speed")); self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Fast (+)", "speed_up", font, "Increase Sim Speed")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, bw*2 + BUTTON_PADDING, bh, "Reset Speed", "speed_reset", font, "Set Speed to 1.0x")); by += bh + BUTTON_PADDING * 2
        self.buttons.append(Button(bx, by, bw, bh, "Grid", "toggle_grid", font, "Toggle Grid View [G]")); self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Quadtree", "toggle_quadtree", font, "Toggle Quadtree View [Q]")); by += bh + BUTTON_PADDING * 2
        self.buttons.append(Button(bx, by, bw, bh, "Brush-", "brush_down", font, "Decrease Brush Size [-]")); self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Brush+", "brush_up", font, "Increase Brush Size [+]")); by += bh + BUTTON_PADDING * 2

        # --- Tools ---
        full_bw = UI_PANEL_WIDTH - 2 * BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Select", ("set_brush", BRUSH_SELECT), font, "Select Agent (Click)")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Delete", ("set_brush", BRUSH_DELETE), font, "Delete Entity (Click/Drag)")); by += bh + BUTTON_PADDING * 2

        # --- Spawning (Water/Threat added) ---
        self.buttons.append(Button(bx, by, full_bw, bh, "Spawn Agent", ("set_brush", BRUSH_SPAWN_AGENT), font, "Click Sim to Spawn Agent")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Spawn Resource", ("set_brush", BRUSH_SPAWN_RESOURCE), font, "Click Sim to Spawn Resource")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Spawn Water Src", ("set_brush", BRUSH_SPAWN_WATER), font, "Click Sim to Spawn Water Source")); by += bh + BUTTON_PADDING # New
        self.buttons.append(Button(bx, by, full_bw, bh, "Spawn Threat", ("set_brush", BRUSH_SPAWN_THREAT), font, "Click Sim to Spawn Threat")); by += bh + BUTTON_PADDING * 2 # New

        # --- Painting (Water added) ---
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Wall", ("set_brush", BRUSH_WALL), font, "Paint Impassable Walls")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Water", ("set_brush", BRUSH_WATER), font, "Paint Water Terrain")); by += bh + BUTTON_PADDING # New
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Clear", ("set_brush", BRUSH_CLEAR), font, "Clear Walls/Water to Plains")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Plains", ("set_brush", BRUSH_PLAINS), font, "Paint Plains Terrain (Cost 1)")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Grass", ("set_brush", BRUSH_GRASS), font, "Paint Grass Terrain (Cost 2)")); by += bh + BUTTON_PADDING
        self.buttons.append(Button(bx, by, full_bw, bh, "Paint Mud", ("set_brush", BRUSH_MUD), font, "Paint Mud Terrain (Cost 5)")); by += bh + BUTTON_PADDING * 2

        # --- Save/Load ---
        self.buttons.append(Button(bx, by, bw, bh, "Save", "save", font, "Save Sim State (F5)")); self.buttons.append(Button(bx + bw + BUTTON_PADDING, by, bw, bh, "Load", "load", font, "Load Sim State (F9)"))

        self._update_button_selected_state()

    def _update_button_selected_state(self):
        """Updates the 'is_selected' state of toggle buttons and tooltips."""
        current_brush = self.environment.paint_brush
        self.active_brush_tooltip = f"Active: {current_brush}"
        font_to_use = self.fonts.get('small', pygame.font.Font(None, UI_FONT_SIZE_SMALL))
        for button in self.buttons:
            selected = False; tooltip_suffix = ""
            if isinstance(button.action, tuple) and button.action[0] == "set_brush":
                if button.action[1] == current_brush:
                    selected = True; self.active_brush_tooltip = f"Active Tool: {button.text}"
                    if button.action[1] in PAINT_BRUSHES: tooltip_suffix = f" (Size: {self.environment.brush_size})"
            button.is_selected = selected
            if button.tooltip:
                full_tooltip = button.tooltip + tooltip_suffix
                try:
                    button.tooltip_surf = font_to_use.render(full_tooltip, True, BUTTON_TEXT_COLOR, (0,0,0))
                    button.tooltip_rect = button.tooltip_surf.get_rect()
                except Exception as e: button.tooltip_surf = None # Disable on error


    def handle_click(self, mouse_pos, game_state):
        """Checks if a UI button was clicked and performs its action."""
        clicked_action = None
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                clicked_action = button.handle_click()
                break

        if clicked_action:
            if clicked_action == "toggle_pause": game_state['paused'] = not game_state['paused']
            elif clicked_action == "play": game_state['paused'] = False
            elif clicked_action == "speed_down": self.environment.speed_multiplier = max(0.1, self.environment.speed_multiplier / 1.5)
            elif clicked_action == "speed_up": self.environment.speed_multiplier = min(10.0, self.environment.speed_multiplier * 1.5)
            elif clicked_action == "speed_reset": self.environment.speed_multiplier = 1.0
            elif clicked_action == "toggle_grid": game_state['draw_grid'] = not game_state['draw_grid']
            elif clicked_action == "toggle_quadtree": game_state['draw_quadtree'] = not game_state['draw_quadtree']
            elif clicked_action == "brush_down":
                self.environment.brush_size = clamp(self.environment.brush_size - 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
                self._update_button_selected_state() # Update tooltips potentially
            elif clicked_action == "brush_up":
                self.environment.brush_size = clamp(self.environment.brush_size + 1, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
                self._update_button_selected_state() # Update tooltips potentially
            elif clicked_action == "save": return "save" # Signal main loop to handle save
            elif clicked_action == "load": return "load" # Signal main loop to handle load
            elif isinstance(clicked_action, tuple) and clicked_action[0] == "set_brush":
                self.environment.paint_brush = clicked_action[1] # Set the new brush
                self._update_button_selected_state() # Update highlights/tooltips
            return True # Indicate UI handled the click
        return False # No UI button clicked

    def draw_minimap(self):
        """Draws the minimap, including water and threats."""
        # 1. Draw Background (Terrain)
        self.minimap_surface.fill(MINIMAP_BG_COLOR)
        for gy in range(0, GRID_HEIGHT, self.map_grid_step_y):
            for gx in range(0, GRID_WIDTH, self.map_grid_step_x):
                terrain_cost = self.environment.grid[gy][gx]
                color = TERRAIN_COLORS.get(terrain_cost, GREY) # Get color for terrain
                map_x = int(gx * GRID_CELL_SIZE * self.map_scale_x)
                map_y = int(gy * GRID_CELL_SIZE * self.map_scale_y)
                map_w = max(1, int(self.map_grid_step_x * GRID_CELL_SIZE * self.map_scale_x))
                map_h = max(1, int(self.map_grid_step_y * GRID_CELL_SIZE * self.map_scale_y))
                pygame.draw.rect(self.minimap_surface, color, (map_x, map_y, map_w, map_h))

        # 2. Draw Entities
        entity_size = 2 # Agent/Resource size
        base_entity_size = 3
        water_entity_size = 3 # Water source size
        threat_entity_size = 2 # Threat marker size

        # Base
        if self.environment.base:
            bx = int(self.environment.base.pos.x * self.map_scale_x); by = int(self.environment.base.pos.y * self.map_scale_y)
            pygame.draw.rect(self.minimap_surface, MINIMAP_BASE_COLOR, (bx - base_entity_size // 2, by - base_entity_size // 2, base_entity_size, base_entity_size))

        # Water Sources
        for water in self.environment.water_sources:
             wx = int(water.pos.x * self.map_scale_x); wy = int(water.pos.y * self.map_scale_y)
             pygame.draw.rect(self.minimap_surface, MINIMAP_WATER_COLOR, (wx - water_entity_size // 2, wy - water_entity_size // 2, water_entity_size, water_entity_size))

        # Resources
        for res in self.environment.resources:
            if res.quantity > 0:
                rx = int(res.pos.x * self.map_scale_x); ry = int(res.pos.y * self.map_scale_y)
                pygame.draw.rect(self.minimap_surface, MINIMAP_RESOURCE_COLOR, (rx, ry, entity_size, entity_size))

        # Threats
        for threat in self.environment.threats:
             tx = int(threat.pos.x * self.map_scale_x); ty = int(threat.pos.y * self.map_scale_y)
             pygame.draw.rect(self.minimap_surface, MINIMAP_THREAT_COLOR, (tx - threat_entity_size // 2, ty - threat_entity_size // 2, threat_entity_size, threat_entity_size))
             # Draw X ? pygame.draw.line(...)

        # Agents (Draw last to be on top)
        for agent in self.environment.agents:
            ax = int(agent.pos.x * self.map_scale_x); ay = int(agent.pos.y * self.map_scale_y)
            agent_map_color = MINIMAP_AGENT_COLLECTOR_COLOR if agent.role == ROLE_COLLECTOR else MINIMAP_AGENT_BUILDER_COLOR
            pygame.draw.rect(self.minimap_surface, agent_map_color, (ax, ay, entity_size, entity_size))


    def draw(self, screen, clock, game_state):
        """Draws the entire UI panel."""
        # --- Panel Background & Separator ---
        ui_panel_rect = pygame.Rect(UI_PANEL_X, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT); pygame.draw.rect(screen, UI_BG_COLOR, ui_panel_rect)
        pygame.draw.line(screen, WHITE, (UI_PANEL_X, 0), (UI_PANEL_X, SCREEN_HEIGHT), 1)

        # --- Draw Minimap ---
        self.draw_minimap(); screen.blit(self.minimap_surface, self.minimap_rect.topleft)
        pygame.draw.rect(screen, MINIMAP_BORDER_COLOR, self.minimap_rect, 1)

        # --- Draw Buttons & Tooltips ---
        mouse_pos = pygame.mouse.get_pos(); hovered_button = None
        for button in self.buttons: button.check_hover(mouse_pos); button.draw(screen)
        if button.is_hovered: hovered_button = button
        if hovered_button: hovered_button.draw_tooltip(screen, mouse_pos)

        # --- Draw Info Text ---
        if self.buttons: info_y_start = self.buttons[-1].rect.bottom + BUTTON_PADDING * 2
        else: info_y_start = MINIMAP_Y + MINIMAP_HEIGHT + 250 # Fallback
        line = 0; font = self.fonts['small']
        def draw_info(text, val=""): # Helper function (same as before)
             nonlocal line; full_text = f"{text}: {val}" if val != "" else text
             try: surf = font.render(full_text, True, WHITE); screen.blit(surf, (UI_PANEL_X + BUTTON_PADDING, info_y_start + line * UI_LINE_HEIGHT)); line += 1
             except Exception as e: print(f"Error rendering UI text '{full_text}': {e}"); line += 1

        # --- General Info ---
        world_time = self.environment.world_time; cycle = DAY_NIGHT_CYCLE_DURATION; time_of_day = world_time % cycle
        time_str = f"{int(world_time // cycle)}d {time_of_day:.1f}s ({'Night' if self.environment.is_night else 'Day'})"
        draw_info("Time", time_str)
        draw_info("FPS", f"{int(clock.get_fps())}")
        draw_info("Speed", f"{self.environment.speed_multiplier:.1f}x {'(PAUSED)' if game_state['paused'] else ''}")

        # --- Agent Counts & Averages ---
        num_agents = len(self.environment.agents); collectors = sum(1 for a in self.environment.agents if a.role == ROLE_COLLECTOR); builders = sum(1 for a in self.environment.agents if a.role == ROLE_BUILDER)
        draw_info("Agents", f"{num_agents} (C:{collectors}, B:{builders})")
        if num_agents > 0:
            avg_energy = sum(a.energy for a in self.environment.agents) / num_agents; avg_hunger = sum(a.hunger for a in self.environment.agents) / num_agents
            avg_thirst = sum(a.thirst for a in self.environment.agents) / num_agents; avg_sleep = sum(a.sleepiness for a in self.environment.agents) / num_agents
            avg_social = sum(a.social for a in self.environment.agents) / num_agents; avg_boredom = sum(a.boredom for a in self.environment.agents) / num_agents
            draw_info("Avg Energy", f"{avg_energy:.1f}"); draw_info("Avg Hunger", f"{avg_hunger:.1f}")
            draw_info("Avg Thirst", f"{avg_thirst:.1f}"); draw_info("Avg Sleepy", f"{avg_sleep:.1f}") # New Averages
            draw_info("Avg Social", f"{avg_social:.1f}"); draw_info("Avg Boredom", f"{avg_boredom:.1f}")
        else: draw_info("Avg Needs", "N/A") # Simpler if no agents

        # --- Resource Info ---
        total_res_quantity = sum(r.quantity for r in self.environment.resources); draw_info("World Res", f"{total_res_quantity} ({len(self.environment.resources)} nodes)")
        draw_info("Base Storage", f"{self.environment.total_base_resources}")

        # --- Tool Info ---
        draw_info("Tool", f"{self.environment.paint_brush}")
        if self.environment.paint_brush in PAINT_BRUSHES: draw_info("Brush Size", f"{self.environment.brush_size}x{self.environment.brush_size}")
        line +=1

        # --- Selected Agent Info (Expanded Needs) ---
        draw_info("Selected Agent:")
        agent = self.environment.selected_agent
        if agent and agent in self.environment.agents:
            state_str = agent.state # Format state string
            if agent.state == STATE_WORKING and agent.action_type: state_str += f" ({agent.action_type} {agent.action_timer:.1f}s)"
            elif agent.state == STATE_SLEEPING: state_str += " (Zzz...)"
            elif agent.current_path: state_str += f" (Path: {agent.path_index+1}/{len(agent.current_path)})"

            draw_info("  ID", agent.id); draw_info("  Role", agent.role)
            draw_info("  State", state_str)
            draw_info("  Energy", f"{agent.energy:.1f}/{MAX_ENERGY}")
            draw_info("  Hunger", f"{agent.hunger:.1f}/{MAX_HUNGER}")
            draw_info("  Thirst", f"{agent.thirst:.1f}/{MAX_THIRST}") # New
            draw_info("  Sleepy", f"{agent.sleepiness:.1f}/{MAX_SLEEPINESS}") # New
            draw_info("  Social", f"{agent.social:.1f}/{MAX_SOCIAL}") # New
            draw_info("  Boredom", f"{agent.boredom:.1f}/{MAX_BOREDOM}") # New
            draw_info("  Fear", f"{agent.fear:.1f}/{MAX_FEAR}") # New
            draw_info("  Carrying", f"{agent.carrying_resource}/{agent.capacity}")

            # Target Info (same logic)
            target_str = "None"
            if agent.target_resource: target_str = f"Res {agent.target_resource.id}"
            elif agent.target_water: target_str = f"Water {agent.target_water.id}" # New
            elif agent.destination_grid_pos:
                 target_str = f"Grid {agent.destination_grid_pos}"
                 # Add context...
            draw_info("  Target", target_str)
        else: draw_info("  None")
        line +=1

        # --- Event Log ---
        draw_info("Event Log:")
        log_font = self.fonts['small']; log_y = info_y_start + line * UI_LINE_HEIGHT
        log_line_h = UI_LINE_HEIGHT - 4; max_log_y = SCREEN_HEIGHT - BUTTON_PADDING
        log_messages = self.environment.get_event_log()
        log_lines_drawn = 0
        for msg in reversed(log_messages[-MAX_EVENT_LOG_MESSAGES:]): # Show last N
            if log_y + log_lines_drawn * log_line_h > max_log_y: break
            try: log_surf = log_font.render(msg, True, LOG_TEXT_COLOR); screen.blit(log_surf, (UI_PANEL_X + BUTTON_PADDING, log_y + log_lines_drawn * log_line_h)); log_lines_drawn += 1
            except Exception as e: continue # Skip render errors