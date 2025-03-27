import pickle
import os
import pygame # Needed for Rect
# Import constants for fallback and Quadtree dimensions
from constants import SIM_WIDTH, SIM_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT

# Need to import classes to allow pickle to reconstruct them when loading
# Ensure all necessary classes are imported
from environment import Environment
from agent import Agent
from entities import Base, Resource, Obstacle
from quadtree import Quadtree
# Import role constants maybe needed by unpickling agents
from constants import ROLE_BUILDER, ROLE_COLLECTOR

SAVE_FILENAME = "agent_sim_save_v5.pkl" # Use consistent version number

def save_simulation(environment, filename=SAVE_FILENAME):
    """Saves the current state of the simulation environment to a file using pickle."""
    print(f"Attempting to save simulation to {filename}...")

    # --- Prepare environment for pickling ---
    # 1. Backup non-serializable or complex attributes
    selected_id_backup = environment.selected_agent.id if environment.selected_agent else None
    quadtree_backup = environment.quadtree # Backup reference if needed immediately after

    # 2. Remove or replace attributes before saving __dict__
    environment.selected_agent = None # Save ID instead of the object reference
    environment.quadtree = None       # Quadtree is rebuilt on load

    # 3. Create data structure to save
    save_data = {
        'environment_state': environment.__dict__, # Save core attributes
        'selected_agent_id': selected_id_backup,   # Save selected agent ID
        # Save class-level ID counters
        'agent_id_counter': Agent._id_counter,
        'resource_id_counter': Resource._id_counter,
    }

    # 4. Perform pickling
    try:
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL) # Use high protocol
        print(f"Simulation saved successfully.")
    except Exception as e:
        print(f"Error saving simulation: {e}")
    finally:
        # --- Restore transient attributes to the original environment object ---
        # Important if the simulation continues running after the save attempt
        environment.quadtree = quadtree_backup # Restore quadtree reference
        # Find agent object from restored ID
        if selected_id_backup is not None:
            environment.selected_agent = None # Clear first
            for agent in environment.agents:
                if agent.id == selected_id_backup:
                    environment.selected_agent = agent
                    break
        # If quadtree was none before, ensure it's recreated if needed now
        if environment.quadtree is None and hasattr(environment, 'sim_width'):
             environment.quadtree = Quadtree(0, pygame.Rect(0, 0, environment.sim_width, environment.sim_height))
             if hasattr(environment, 'rebuild_quadtree'):
                 environment.rebuild_quadtree()


def load_simulation(filename=SAVE_FILENAME):
    """Loads a simulation state from a file and returns a new environment object."""
    print(f"Attempting to load simulation from {filename}...")
    if not os.path.exists(filename):
        print(f"Error: Save file '{filename}' not found.")
        return None, None # Return None for environment and selected ID

    try:
        # Load the data from the pickle file
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # --- Recreate the environment object ---
        # 1. Get necessary dimensions from saved state (use constants as fallback)
        env_state = save_data['environment_state']
        width = env_state.get('width', SCREEN_WIDTH)
        height = env_state.get('height', SCREEN_HEIGHT)
        sim_width = env_state.get('sim_width', SIM_WIDTH)
        sim_height = env_state.get('sim_height', SIM_HEIGHT)

        # 2. Create a new Environment instance
        loaded_env = Environment(width, height, sim_width, sim_height)

        # 3. Update the new environment's dictionary with the loaded state
        # This restores entity lists, time, grid, etc.
        loaded_env.__dict__.update(env_state)

        # 4. Restore class-level ID counters from save data
        Agent._id_counter = save_data.get('agent_id_counter', Agent._id_counter)
        Resource._id_counter = save_data.get('resource_id_counter', Resource._id_counter)

        # 5. Get the ID of the agent that was selected when saved
        selected_id = save_data['selected_agent_id']

        # --- Post-load setup (Recreate transient objects) ---
        # a) Recreate Quadtree (needs to happen AFTER __dict__.update)
        loaded_env.quadtree = Quadtree(0, pygame.Rect(0, 0, loaded_env.sim_width, loaded_env.sim_height))
        # b) Populate Quadtree with loaded entities
        if hasattr(loaded_env, 'rebuild_quadtree'):
            loaded_env.rebuild_quadtree()
        else:
            print("Warning: Loaded environment missing rebuild_quadtree method.")

        # c) UI Manager and Fonts are handled externally in main.py after this function returns

        print(f"Simulation loaded successfully.")
        # Return the new environment and the selected agent's ID
        return loaded_env, selected_id
    except ModuleNotFoundError as e:
         print(f"Error loading simulation: Module not found. Class definitions might have changed or are missing. Error: {e}")
         return None, None
    except AttributeError as e:
         print(f"Error loading simulation: Attribute error. Save file might be incompatible with current code. Error: {e}")
         return None, None
    except Exception as e:
        print(f"Error loading simulation: An unexpected error occurred. Error: {e}")
        # Consider printing traceback for detailed debugging
        # import traceback
        # traceback.print_exc()
        return None, None # Return None on error