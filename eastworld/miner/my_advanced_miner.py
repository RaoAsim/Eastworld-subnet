import bittensor as bt
from collections import deque
from typing import TypedDict
import operator
import os

# Base classes and protocol
from eastworld.base.miner import BaseMiner
from eastworld.protocol import Observation

# LangGraph state machine
from langgraph.graph import StateGraph, END

# SLAM and Memory modules
from eastworld.miner.slam.isam import ISAM2 as ISAM
from eastworld.miner.memory import JSONFileMemory

# ### REASONING ###
# Import our new prompt loader
from eastworld.miner.prompts import load_prompt
# Import LangChain components to interact with an LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

class AgentState(TypedDict):
    observation: Observation
    plan: list[str]
    goals: list[str]
    reflection: str
    action: str
    action_log: deque[str]
    slam: ISAM

class MyAdvancedAgent(BaseMiner):
    """
    The final version of our agent: state machine, SLAM, persistent memory,
    and an LLM-powered cognitive cycle.
    """
    def __init__(self):
        super().__init__()
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

        # ### REASONING ###
        # Initialize the LLM. You must have OPENAI_API_KEY set in your environment.
        # We use a powerful model for reasoning.
        try:
            self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
        except Exception as e:
            bt.logging.error(f"Failed to initialize LLM. Make sure OPENAI_API_KEY is set. Error: {e}")
            # Exit if LLM is not available, as the agent cannot function.
            exit(1)

        # Load the specialized prompts
        self.objective_prompt = load_prompt('senior_objective_reevaluation')
        self.action_prompt = load_prompt('senior_action_selection')
        self.review_prompt = load_prompt('senior_after_action_review')

        # Create LangChain "chains" for each cognitive step.
        # A chain combines a prompt, a model, and an output parser.
        self.objective_chain = ChatPromptTemplate.from_template(self.objective_prompt) | self.llm | JsonOutputParser()
        self.action_chain = ChatPromptTemplate.from_template(self.action_prompt) | self.llm | JsonOutputParser()
        self.review_chain = ChatPromptTemplate.from_template(self.review_prompt) | self.llm | StrOutputParser()
        
        # Initialize Memory and SLAM
        self.goals = ["Explore the crashed spacecraft and identify the needs of the survivors."]
        self.plan = []

        self.memory = JSONFileMemory(filepath=os.path.join(self.config.full_path, "agent_metadata.json"))

        loaded_memory = self.memory.load()
        if loaded_memory:
            self.goals = loaded_memory.get("goals", self.goals)
            self.plan = loaded_memory.get("plan", self.plan)


        # Define the State Machine Graph
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
        bt.logging.info("🧐 Reviewing last action with LLM...")
        try:
            reflection = self.review_chain.invoke({
                "goals": self.goals,
                "plan": self.plan,
                "action": state['action'],
                "outcome": state['observation'].action_log[-1]
            })
            bt.logging.info(f"LLM reflection: {reflection}")
        except Exception as e:
            bt.logging.error(f"LLM call failed in action review: {e}")
            reflection = "Reflection failed due to an error."
        state['reflection'] = reflection
        return state



    async def forward(self, observation: Observation) -> str:
        bt.logging.info("Forward call received, invoking full cognitive cycle.")
        
        initial_state: AgentState = {
            "observation": observation,
            "plan": self.plan,
            "goals": self.goals,
            "reflection": "",
            "action": "",
            "action_log": deque(observation.action_log, maxlen=50),
            # ### SLAM ###
            # Pass our SLAM instance into the state.
            "slam": self.slam,
        }

        final_state = self.app.invoke(initial_state)
        return final_state['action']