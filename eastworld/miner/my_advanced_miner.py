import bittensor as bt
import json
from collections import deque
from typing import TypedDict
import os

# Base classes and protocol
from eastworld.base.miner import BaseMinerNeuron
from eastworld.protocol import Observation

# LangGraph state machine
from langgraph.graph import StateGraph, END

# SLAM and Memory modules
# Assuming ISAM2 is the correct class name from your custom SLAM file
from eastworld.miner.slam.isam import ISAM2 as ISAM 
from eastworld.miner.memory import JSONFileMemory
from pathlib import Path
# REASONING
from eastworld.miner.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# --- AgentState Definition ---
# I've fixed the indentation here, which can cause syntax errors.
class AgentState(TypedDict):
    observation: Observation
    plan: list[str]
    goals: list[str]
    reflection: str
    action: str
    action_log: deque[str]
    slam: ISAM

class MyAdvancedAgent(BaseMinerNeuron):
    """
    The final, corrected version of our agent: state machine, SLAM, persistent memory,
    and an LLM-powered cognitive cycle.
    """
    def __init__(self):
        super().__init__()
        
        # --- SLAM Persistence ---
        script_dir = Path(__file__).parent.resolve()
        
        slam_data_path = script_dir / "slam_data"
        metadata_filepath = script_dir / "agent_metadata.json" # Use this for metadata
        
        bt.logging.info(f"Using portable path for SLAM data: {slam_data_path}")
        bt.logging.info(f"Using portable path for metadata: {metadata_filepath}")
        
        # --- SLAM Persistence ---
        # Using the portable path for SLAM is correct.
        self.slam = ISAM(data_dir=str(slam_data_path))
        bt.logging.info("SLAM module initialized.")
        try:
            self.slam.load(load_path=str(slam_data_path))
        except FileNotFoundError:
             bt.logging.warning("SLAM data directory not found. Starting with a fresh SLAM map.")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred loading SLAM data: {e}")

        # --- LLM and Prompts Initialization ---
        # This section is correct.
        try:
            self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
        except Exception as e:
            bt.logging.error(f"Failed to initialize LLM. Make sure OPENAI_API_KEY is set. Error: {e}")
            exit(1)

        self.objective_prompt = load_prompt('senior_objective_reevaluation')
        self.action_prompt = load_prompt('senior_action_selection')
        self.review_prompt = load_prompt('senior_after_action_review')

        self.objective_chain = ChatPromptTemplate.from_template(self.objective_prompt) | self.llm | JsonOutputParser()
        self.action_chain = ChatPromptTemplate.from_template(self.action_prompt) | self.llm | JsonOutputParser()
        self.review_chain = ChatPromptTemplate.from_template(self.review_prompt) | self.llm | StrOutputParser()
        
        # --- Metadata Persistence ---
        # This logic is also correct and now uses the fixed memory class.
        self.goals = ["Explore the crashed spacecraft and identify the needs of the survivors."]
        self.plan = []
        self.memory = JSONFileMemory(filepath=str(metadata_filepath))
        loaded_memory = self.memory.load()
        if loaded_memory:
            self.goals = loaded_memory.get("goals", self.goals)
            self.plan = loaded_memory.get("plan", self.plan)

        # --- State Machine Definition ---
        # The graph structure is correct.
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
        bt.logging.info("Fully intelligent agent compiled and ready.")

    # --- Graph Nodes ---
    
    def update_map(self, state: AgentState) -> AgentState:
        bt.logging.info("🗺️  Running SLAM iteration...")
        obs = state['observation']
        
        # ### CORRECTED ###
        # The original odometry parsing was brittle. This is a more robust way to handle it.
        # We check if the last action was actually a move action before trying to parse it.
        try:
            last_action_str = obs.action_log[-1] if obs.action_log else ""
            if "Moved" in last_action_str:
                # This parsing logic is still simple, but now it only runs when it should.
                parts = last_action_str.replace("Moved ", "").replace("cm to the ", " ").split()
                distance = float(parts[0])
                direction = parts[1]
                # Assuming your ISAM2 class has this method signature.
                self.slam.run_iteration(obs.lidar, distance, direction)
            else:
                bt.logging.info("Last action was not a move, skipping SLAM odometry update.")
        except Exception as e:
            bt.logging.error(f"Could not parse odometry or run SLAM iteration: {e}")
            # Good fallback logic to still update the map with sensor data.
            self.slam._update_grid_map(self.slam.get_current_pose(), obs.lidar)
            
        return state

    def objective_reevaluation(self, state: AgentState) -> AgentState:
        # This node was already correct. No changes needed.
        bt.logging.info("🔍 Re-evaluating objectives with LLM...")
        try:
            response = self.objective_chain.invoke({
                "goals": self.goals,
                "plan": self.plan,
                "observation": state['observation'].perception,
            })
            self.goals = response.get("goals", self.goals)
            self.plan = response.get("plan", self.plan)
            bt.logging.info(f"LLM updated goals: {self.goals}")
        except Exception as e:
            bt.logging.error(f"LLM call failed in objective re-evaluation: {e}")
        state['goals'] = self.goals
        state['plan'] = self.plan
        return state

    def action_selection(self, state: AgentState) -> AgentState:
        # ### CRITICAL FIX ###
        # This node was using placeholder logic. It has been restored to use the LLM
        # for intelligent action selection. This connects the agent's brain.
        bt.logging.info("🤔 Selecting action with LLM...")
        try:
            current_x, current_y, _ = self.slam.get_current_pose()
            agent_pose_str = f"({current_x:.2f}, {current_y:.2f})"

            response = self.action_chain.invoke({
                "goals": self.goals,
                "plan": self.plan,
                "agent_pose": agent_pose_str,
                "observation": state['observation'].perception,
                "inventory": str(state['observation'].inventory),
                "available_actions": str(state['observation'].available_actions),
                "action_log": "\n".join(state['action_log'])
            })
            chosen_action = json.dumps(response) # Ensure the output is a valid JSON string
            bt.logging.info(f"LLM chose action: {chosen_action}")
        except Exception as e:
            bt.logging.error(f"LLM call failed in action selection: {e}")
            chosen_action = '{"tool_name": "move_forward", "tool_args": {}}' # Fallback
            
        state["action"] = chosen_action
        return state
        
    def after_action_review(self, state: AgentState) -> AgentState:
        # This node was correct. No changes needed.
        bt.logging.info("🧐 Reviewing last action with LLM...")
        try:
            reflection = self.review_chain.invoke({
                "goals": self.goals,
                "plan": self.plan,
                "action": state['action'],
                "outcome": state['observation'].action_log[-1] if state['observation'].action_log else "N/A"
            })
            bt.logging.info(f"LLM reflection: {reflection}")
        except Exception as e:
            bt.logging.error(f"LLM call failed in action review: {e}")
            reflection = "Reflection failed due to an error."
        state['reflection'] = reflection
        return state

    def save_memory(self, state: AgentState) -> AgentState:
        bt.logging.info("💾 Saving all memories...")
        # 1. Save SLAM data using its own method.
        self.slam.save(save_path=self.slam.data_dir)
        # 2. Save metadata using the corrected memory class.
        self.memory.save(goals=self.goals, plan=self.plan)
        return state
    
    async def forward(self, observation: Observation) -> str:
        bt.logging.info("Forward call received, invoking full cognitive cycle.")
        
        # This initialization logic is correct.
        initial_state: AgentState = {
            "observation": observation,
            "plan": self.plan,
            "goals": self.goals,
            "reflection": "",
            "action": "",
            "action_log": deque(observation.action_log, maxlen=50),
            "slam": self.slam,
        }

        final_state = self.app.invoke(initial_state)
        return final_state['action']

