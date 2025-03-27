import pickle
import os
import pygame # Needed for Quadtree Rect potentially if saved directly
from constants import SIM_WIDTH, SIM_HEIGHT # Needed for Quadtree bounds

# Need to import classes to allow pickle to reconstruct them when loading
from environment import Environment
from agent import Agent
from entities import Base, Resource, Obstacle
from quadtree import Quadtree
# Import role constants in case they are needed by unpickling agent roles
from constants import ROLE_BUILDER, ROLE_COLLECTOR ,SCREEN_WIDTH, SCREEN_HEIGHT


SAVE_FILENAME = "agent_sim_save_v5.pkl" # Incremented version number

def save_simulation(environment, filename=SAVE_FILENAME):
    """Saves the current state of the simulation environment to a file using pickle."""
    print(f"Attempting to save simulation to {filename}...")

    # --- Prepare environment for pickling ---
    # 1. Backup non-serializable or complex attributes
    selected_id_backup = environment.selected_agent.id if environment.selected_agent else None
    # The UI Manager is not saved, it's recreated on load based on the loaded environment

    # 2. Remove or replace non-serializable/recreatable attributes before saving __dict__
    #    Quadtree is rebuilt on load, no need to save its complex structure.
    quadtree_backup = environment.quadtree # Keep backup if needed immediately after
    environment.selected_agent = None # Save ID instead of reference
    environment.quadtree = None       # Don't save quadtree

    # 3. Create data structure to save
    save_data = {
        'environment_state': environment.__dict__, # Save the core attributes dictionary
        'selected_agent_id': selected_id_backup,   # Save the ID of the selected agent
        # Save class-level ID counters to preserve uniqueness across loads
        'agent_id_counter': Agent._id_counter,
        'resource_id_counter': Resource._id_counter,
    }

    # 4. Perform pickling
    try:
        with open(filename, 'wb') as f:
            # Use a high protocol for efficiency if compatibility isn't an issue
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
        print(f"Simulation saved successfully.")
    except Exception as e:
        print(f"Error saving simulation: {e}")
    finally:
        # --- Restore transient attributes to the original environment object ---
        # This is crucial if the simulation continues running immediately after saving
        environment.quadtree = quadtree_backup # Restore quadtree reference
        # Find agent object from restored ID
        if selected_id_backup is not None:
            for agent in environment.agents:
                if agent.id == selected_id_backup:
                    environment.selected_agent = agent
                    break
        # Note: UI Manager doesn't need restoring here, as it's managed externally

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
        # 1. Create a new Environment instance (dimensions are needed for init)
        env_state = save_data['environment_state']
        # Check for necessary keys from saved state for initialization
        width = env_state.get('width', SCREEN_WIDTH) # Use constants as fallback
        height = env_state.get('height', SCREEN_HEIGHT)
        sim_width = env_state.get('sim_width', SIM_WIDTH)
        sim_height = env_state.get('sim_height', SIM_HEIGHT)
        loaded_env = Environment(width, height, sim_width, sim_height)

        # 2. Update the new environment's dictionary with the loaded state
        # This overwrites default values with the saved ones (including lists of entities, time, etc.)
        loaded_env.__dict__.update(env_state)

        # 3. Restore class-level ID counters from save data
        # Use saved value, or keep current class value if key missing in old save files
        Agent._id_counter = save_data.get('agent_id_counter', Agent._id_counter)
        Resource._id_counter = save_data.get('resource_id_counter', Resource._id_counter)

        # 4. Get the ID of the agent that was selected when saved
        selected_id = save_data['selected_agent_id']

        # --- Post-load setup (Recreate transient objects) ---
        # a) Recreate Quadtree: Needs to be done AFTER entity lists are loaded
        loaded_env.quadtree = Quadtree(0, pygame.Rect(0, 0, loaded_env.sim_width, loaded_env.sim_height))
        # b) Populate Quadtree: Call rebuild AFTER entities are loaded into lists
        if hasattr(loaded_env, 'rebuild_quadtree'):
            loaded_env.rebuild_quadtree()
        else:
            print("Warning: Loaded environment missing rebuild_quadtree method.")

        # c) UI Manager and Fonts are handled externally in main.py after load

        print(f"Simulation loaded successfully.")
        # Return the newly created environment and the selected agent's ID
        return loaded_env, selected_id
    except Exception as e:
        print(f"Error loading simulation: {e}")
        # Can add more specific error handling (e.g., for ModuleNotFoundError if classes changed)
        return None, None # Return None on error