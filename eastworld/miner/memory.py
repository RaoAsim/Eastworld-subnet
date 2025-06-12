import bittensor as bt
import json
import os

# No longer need to import ISAM here

class JSONFileMemory:
    """
    Handles saving and loading the agent's METADATA (goals, plans) to a JSON file.
    SLAM persistence is handled separately by the ISAM2 class itself.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Ensure the directory for the memory file exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bt.logging.info(f"Metadata memory module initialized. Using file: {self.filepath}")

    def save(self, goals: list[str], plan: list[str]):
        """Saves the agent's metadata to the JSON file."""
        try:
            # The state now only contains metadata, not the SLAM object.
            state_to_save = {
                "goals": goals,
                "plan": plan,
            }
            with open(self.filepath, "w") as f:
                json.dump(state_to_save, f, indent=4)
            bt.logging.info(f"Agent metadata saved successfully to {self.filepath}")
        except Exception as e:
            bt.logging.error(f"Error saving agent metadata: {e}")

    def load(self) -> dict | None:
        """Loads the agent's metadata from the JSON file."""
        if not os.path.exists(self.filepath):
            bt.logging.warning("Metadata memory file not found. Starting with fresh metadata.")
            return None
        
        try:
            with open(self.filepath, "r") as f:
                loaded_state = json.load(f)
            bt.logging.info(f"Agent metadata loaded successfully from {self.filepath}")
            return loaded_state
        except Exception as e:
            bt.logging.error(f"Error loading metadata: {e}")
            return None