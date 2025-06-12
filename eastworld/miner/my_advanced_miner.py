import bittensor as bt
from collections import deque
from typing import TypedDict


from eastworld.base.miner import BaseMinerNeuron
from eastworld.protocol import Observation

# Import LangGraph to build our state machine
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    observation: Observation       # The raw data from the validator
    plan: list[str]                # The agent's long-term plan
    goals: list[str]               # The agent's high-level goals
    reflection: str                # The agent's reflection on its last action
    action: str                    # The chosen action to execute
    action_log: deque[str]         # A log of recent actions and their outcomes

class MyAdvancedAgent(BaseMinerNeuron):
    """
    This is our advanced agent. It uses a LangGraph state machine to drive its behavior,
    making it more robust and intelligent than a simple loop-based agent.
    """
    def __init__(self):
        super().__init__() # Make sure to call the parent class constructor

        # === Define the State Machine Graph ===
        workflow = StateGraph(AgentState)

        # 1. Define the Nodes (the "states" or "steps" in our agent's thinking)
        workflow.add_node("objective_reevaluation", self.objective_reevaluation)
        workflow.add_node("action_selection", self.action_selection)
        workflow.add_node("after_action_review", self.after_action_review)

        # 2. Define the Edges (the "transitions" between states)
        workflow.set_entry_point("objective_reevaluation") # The cycle starts here
        workflow.add_edge("objective_reevaluation", "action_selection")
        workflow.add_edge("action_selection", "after_action_review")
        workflow.add_edge("after_action_review", END) # The cycle ends here for one turn

        # 3. Compile the graph
        self.app = workflow.compile()
        bt.logging.info("Advanced agent state machine compiled successfully.")

    # --- These methods are the nodes of our graph ---

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
        """
        Node 2: Based on the goals and current observation, choose the next action.
        """
        bt.logging.info("🤔 Selecting action...")
        # This is where the core "thinking" happens.
        # It should look at state['observation'] and state['goals'] to choose an action.
        # We will implement the advanced logic for this in a later step.
        # For now, we'll choose a simple default action like "move_forward".
        
        # A simple check: if a tool is available, use it. Otherwise, move.
        available_actions = state['observation'].available_actions
        chosen_action = '{"tool_name": "move_forward", "tool_args": {}}' # Default action
        
        if available_actions:
            # A more advanced agent would use an LLM to pick the best action.
            # We'll just pick the first one for simplicity.
            chosen_action = available_actions[0]
        state['action'] = chosen_action
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
            "goals": ["Explore the area and survive."], # Example goal
            "reflection": "",
            "action": "",
             "action_log": deque(observation.action_log, maxlen=50),
        }

        # 2. Run the state machine
        final_state = self.app.invoke(initial_state)

        # 3. Return the action chosen by the "action_selection" node
        bt.logging.info(f"State machine finished. Chosen action: {final_state['action']}")
        return final_state['action']