import bittensor as bt
from collections import deque
from typing import TypedDict

import os
from eastworld.base.miner import BaseMinerNeuron
from eastworld.protocol import Observation
from eastworld.miner.slam.isam import ISAM2 as ISAM
# Import LangGraph to build our state machine
from langgraph.graph import StateGraph, END
from eastworld.miner.memory import JSONFileMemory

class AgentState(TypedDict):
    observation: Observation       # The raw data from the validator
    plan: list[str]                # The agent's long-term plan
    goals: list[str]               # The agent's high-level goals
    reflection: str                # The agent's reflection on its last action
    action: str                    # The chosen action to execute
    action_log: deque[str]
    slam: ISAM       

class MyAdvancedAgent(BaseMinerNeuron):
    """
    This is our advanced agent. It uses a LangGraph state machine to drive its behavior,
    making it more robust and intelligent than a simple loop-based agent.
    """
    def __init__(self):
        super().__init__() # Make sure to call the parent class constructor
        slam_data_path = os.path.join(self.config.full_path, "slam_data")

        self.slam = ISAM(data_dir=slam_data_path)
        bt.logging.info("SLAM module initialized.")
        try:
            self.slam.load(load_path=slam_data_path)
        except FileNotFoundError:
             bt.logging.warning("SLAM data directory not found. Starting with a fresh SLAM map.")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred loading SLAM data: {e}")


        # --- Metadata Persistence ---
        # 1. Initialize the memory module for goals and plans.
        self.memory = JSONFileMemory(filepath=os.path.join(self.config.full_path, "agent_metadata.json"))
        # 2. Load the metadata.
        self.goals = ["Explore the area and survive."]
        self.plan = []
        loaded_metadata = self.memory.load()
        if loaded_metadata:
            self.goals = loaded_metadata.get("goals", self.goals)
            self.plan = loaded_metadata.get("plan", self.plan)

        workflow = StateGraph(AgentState)
        workflow.add_node("update_map", self.update_map)
        workflow.add_node("objective_reevaluation", self.objective_reevaluation)
        workflow.add_node("action_selection", self.action_selection)
        workflow.add_node("after_action_review", self.after_action_review)
        workflow.add_node("save_memory", self.save_memory)
        workflow.set_entry_point("update_map")
        workflow.add_edge("update_map", "objective_reevaluation")
        workflow.add_edge("objective_reevaluation", "action_selection")
        workflow.add_edge("action_selection", "after_action_review")
        workflow.add_edge("after_action_review", "save_memory")
        workflow.add_edge("save_memory", END)
        self.app = workflow.compile()
        bt.logging.info("Advanced agent with corrected persistent memory compiled.")

    # --- These methods are the nodes of our graph ---
    def update_map(self, state: AgentState) -> AgentState:
        """
        ### MODIFIED ###
        The ISAM2 class has a `run_iteration` method that does everything.
        It updates the pose, updates GTSAM, and updates the grid map.
        We should use that instead of calling update_map directly.
        """
        bt.logging.info("🗺️ Running SLAM iteration...")
        obs = state['observation']
        # The run_iteration method handles odometry and lidar processing.
        # We need to extract the distance and direction from the odometry log.
        # This is a placeholder, as the exact format of odometry_log isn't defined in protocol.py
        # Let's assume the log is like: "Moved 100cm to the north."
        try:
            last_move = obs.action_log[-1] if obs.action_log else "Moved 0cm to the east"
            parts = last_move.replace("Moved ", "").replace("cm to the ", " ").split()
            distance = float(parts[0])
            direction = parts[1]
            self.slam.run_iteration(obs.lidar, distance, direction)
        except Exception as e:
            bt.logging.error(f"Could not parse odometry or run SLAM iteration: {e}")
            # Fallback to just updating the map if run_iteration fails
            self.slam._update_grid_map(self.slam.current_pose, obs.lidar)
            
        return state
    
    def objective_reevaluation(self, state: AgentState) -> AgentState:
        """
        Node 1: Look at the current situation and decide if the goals are still valid.
        For now, we'll keep it simple. In the future, this will use an LLM.
        """
        bt.logging.info("🔍 Re-evaluating objectives...")
        # In a real implementation, you would use an LLM with a specific prompt here.
        # For now, we'll just pass the state through.
        return state

    def action_selection(self, state: AgentState) -> AgentState:
        current_x, current_y, current_theta = self.slam.get_current_pose()
        bt.logging.info(f"🤔 Selecting action... Current pose: ({current_x:.2f}, {current_y:.2f})")
        available_actions = state['observation'].available_actions
        chosen_action = '{"tool_name": "move_forward", "tool_args": {}}'
        if available_actions:
            chosen_action = available_actions[0]
        state["action"]=chosen_action
        return state
      

    def save_memory(self, state: AgentState) -> AgentState:
        """### MODIFIED ###
        This node now orchestrates two separate save operations.
        """
        bt.logging.info("💾 Saving all memories...")
        # 1. Tell the SLAM module to save itself to its dedicated directory.
        self.slam.save(save_path=self.slam.data_dir)
        # 2. Tell the metadata memory module to save the other info.
        self.memory.save(goals=self.goals, plan=self.plan)
        return state
    
    def after_action_review(self, state: AgentState) -> AgentState:
        """
        Node 3: Review the result of the last action and reflect on it.
        """
        bt.logging.info("🧐 Reviewing last action...")
        last_action_result = state['observation'].action_log[-1] # Get the most recent log
        
        # Here you would use an LLM with a prompt like "senior_after_action_review.txt"
        # to generate a reflection.
        reflection = f"I just did '{state['action']}' and the result was '{last_action_result}'. I should continue with the plan."
        
        state['reflection'] = reflection
        return state


    # This is the main entry point called by the Bittensor network
    async def forward(self, observation: Observation) -> str:
        """
        This function is called by the validator. It receives the observation,
        runs it through our state machine, and returns the chosen action.
        """
        bt.logging.info("Forward call received, invoking state machine.")
        
        # 1. Initialize the state for this run

        initial_state: AgentState = {
            "observation": observation,
            "plan": [],
            "goals": ["Explore the area and survive."],
            "reflection": "",
            "action": "",
            "action_log": deque(observation.action_log, maxlen=50),
            # ### SLAM ###
            # Pass our SLAM instance into the state.
            "slam": self.slam,
        }

        # 2. Run the state machine
        final_state = self.app.invoke(initial_state)

        # 3. Return the action chosen by the "action_selection" node
        bt.logging.info(f"State machine finished. Chosen action: {final_state['action']}")
        return final_state['action']