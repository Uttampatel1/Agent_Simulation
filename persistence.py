import pickle
import os
import pygame
from constants import SIM_WIDTH, SIM_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT

# Import ALL classes that might be stored in the environment's lists
from environment import Environment
from agent import Agent
from entities import Base, Resource, Obstacle, WaterSource, Threat # Added WaterSource, Threat
from quadtree import Quadtree
from constants import ROLE_BUILDER, ROLE_COLLECTOR # Role constants

SAVE_FILENAME = "agent_sim_save_v6.pkl" # Increment version

def save_simulation(environment, filename=SAVE_FILENAME):
    # ... (Save logic remains structurally the same) ...
    print(f"Attempting to save simulation to {filename}...")
    selected_id_backup = environment.selected_agent.id if environment.selected_agent else None
    quadtree_backup = environment.quadtree
    environment.selected_agent = None; environment.quadtree = None
    save_data = {
        'environment_state': environment.__dict__, 'selected_agent_id': selected_id_backup,
        'agent_id_counter': Agent._id_counter, 'resource_id_counter': Resource._id_counter,
        'water_id_counter': WaterSource._id_counter, 'threat_id_counter': Threat._id_counter, # Save new counters
    }
    try:
        with open(filename, 'wb') as f: pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
        print(f"Simulation saved successfully.")
    except Exception as e: print(f"Error saving simulation: {e}")
    finally: # Restore
        environment.quadtree = quadtree_backup
        if selected_id_backup is not None:
            environment.selected_agent = None
            for agent in environment.agents:
                if agent.id == selected_id_backup: environment.selected_agent = agent; break
        if environment.quadtree is None and hasattr(environment, 'sim_width'): # Recreate if None
             environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
             if hasattr(environment, 'rebuild_quadtree'): environment.rebuild_quadtree()


def load_simulation(filename=SAVE_FILENAME):
    # ... (Load logic remains structurally the same, relies on classes being available) ...
    print(f"Attempting to load simulation from {filename}...")
    if not os.path.exists(filename): print(f"Error: Save file '{filename}' not found."); return None, None
    try:
        with open(filename, 'rb') as f: save_data = pickle.load(f)
        env_state = save_data['environment_state']
        width = env_state.get('width', SCREEN_WIDTH); height = env_state.get('height', SCREEN_HEIGHT)
        sim_width = env_state.get('sim_width', SIM_WIDTH); sim_height = env_state.get('sim_height', SIM_HEIGHT)
        loaded_env = Environment(width, height, sim_width, sim_height)
        loaded_env.__dict__.update(env_state)

        # Restore class counters
        Agent._id_counter = save_data.get('agent_id_counter', Agent._id_counter)
        Resource._id_counter = save_data.get('resource_id_counter', Resource._id_counter)
        WaterSource._id_counter = save_data.get('water_id_counter', WaterSource._id_counter) # Restore new counters
        Threat._id_counter = save_data.get('threat_id_counter', Threat._id_counter)         # Restore new counters

        selected_id = save_data['selected_agent_id']

        # Rebuild quadtree AFTER loading state
        loaded_env.quadtree = Quadtree(0, pygame.Rect(0, 0, loaded_env.sim_width, loaded_env.sim_height))
        if hasattr(loaded_env, 'rebuild_quadtree'): loaded_env.rebuild_quadtree()

        print(f"Simulation loaded successfully.")
        return loaded_env, selected_id
    except ModuleNotFoundError as e: print(f"Error loading: Module not found. {e}"); return None, None
    except AttributeError as e: print(f"Error loading: Attribute error (incompatible save?). {e}"); return None, None
    except Exception as e: print(f"Error loading simulation: {e}"); return None, None