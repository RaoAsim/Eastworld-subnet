import os
from pathlib import Path

def load_prompt(name: str) -> str:
    """
    Loads a prompt from the 'prompts' directory.
    """
    # Assuming this script is in eastworld/miner, prompts are in eastworld/miner/prompts
    base_dir = Path(__file__).parent
    prompt_path = base_dir / "prompts" / f"{name}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, "r") as f:
        return f.read()