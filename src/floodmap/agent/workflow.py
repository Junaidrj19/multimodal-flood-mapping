from langgraph.graph import StateGraph, END
from floodmap.agent.state import AgentState

# --- TEAMMATE'S REAL DECISION ENGINE IMPORT ---
try:
    from decision_logic import prioritize_settlements
except ImportError:
    def prioritize_settlements(settlements):
        print("⚠️ Warning: decision_logic could not be imported. Using fallback.")
        return [{"settlement_id": "V1", "score": 920, "priority": "HIGH", "reasons": ["Fallback"]}]

def retrieve_evidence_node(state: AgentState):
    """Step 1: Retrieve candidate settlements matching your teammate's required schema."""
    question = state.get("question", "")
    print(f"🤖 Agent processing query: {question}")
    
    # ✨ Formatted with the exact keys her evaluate_settlement function looks for!
    settlements_data = [
        {
            "settlement_id": "Bridge_B7",
            "name": "Downstream Settlement V1",
            "connectivity": "POTENTIALLY_CUT_OFF",
            "flood_exposure": "HIGH",
            "estimated_people_needing_help": 150,
            "data_confidence": "MEDIUM"
        },
        {
            "settlement_id": "Village_V2",
            "name": "Riverside Community V2",
            "connectivity": "CONNECTED",
            "flood_exposure": "MEDIUM",
            "estimated_people_needing_help": 40,
            "data_confidence": "HIGH"
        }
    ]
    return {"evidence_nodes": settlements_data}

def run_decision_engine_node(state: AgentState):
    """Step 2: Pass structured data directly to your teammate's decision_logic.py engine."""
    settlements = state.get("evidence_nodes", [])
    
    # Calls her real prioritization engine!
    ranked_outcomes = prioritize_settlements(settlements)
    
    return {"simulated_outcomes": ranked_outcomes}

def format_output_node(state: AgentState):
    """Step 3: Structure the final prioritized recommendations for the workflow output."""
    outcomes = state.get("simulated_outcomes", [])
    best_outcome = outcomes[0] if outcomes else {}
    
    recommendation = {
        "primary_action": best_outcome,
        "alternatives": outcomes[1:] if len(outcomes) > 1 else []
    }
    return {"final_recommendation": recommendation}

# --- BUILD THE LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("retrieve_evidence", retrieve_evidence_node)
workflow.add_node("run_decision_engine", run_decision_engine_node)
workflow.add_node("format_output", format_output_node)

# Define the execution flow
workflow.set_entry_point("retrieve_evidence")
workflow.add_edge("retrieve_evidence", "run_decision_engine")
workflow.add_edge("run_decision_engine", "format_output")
workflow.add_edge("format_output", END)

# Compile the workflow application
agent_app = workflow.compile()