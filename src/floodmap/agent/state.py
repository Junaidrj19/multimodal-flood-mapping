from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str                         # e.g., "Which settlements need attention first?"
    evidence_nodes: List[Dict[str, Any]]  # Evidence fetched from the graph
    simulated_outcomes: List[Dict[str, Any]] # Results from the team's decision engine
    final_recommendation: Optional[Dict[str, Any]] # Final prioritized response