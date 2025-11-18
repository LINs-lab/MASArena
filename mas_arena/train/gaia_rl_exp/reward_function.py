"""
Composite reward function that interacts with agent memory.

The total reward is calculated as:
R_total = R_base + w1 * R_innovation - w2 * R_repetition - w3 * R_efficiency
"""
from typing import Dict, List, Any

# ==============================================================================
# 1. Trajectory Parsing and Strategy Extraction
# ==============================================================================

def _parse_structured_trajectory(trajectory: List[Dict]) -> Dict[str, Any]:
    """
    Parses the complex, hierarchical trajectory from smolagents into a structured
    summary of key strategic decisions.

    This is the most critical pre-processing step for the reward function. It
    distinguishes between the manager's high-level plan and the specialists'
    execution details.

    Args:
        trajectory: The raw list of message dictionaries from the agent loop.

    Returns:
        A dictionary containing structured information, for example:
        {
            "manager_decisions": [
                ("call_tool", "search_agent", "{'query': '...'}"),
                ("call_tool", "PythonInterpreter", "{'code': '...'}"),
                ...
            ],
            "final_answer": "The final answer string from the agent.",
            "total_steps": 15,
            "error_steps": 2,
            "agent_trajectory": {
                "manager_agent": [...],
                "search_agent": [...]
            }
        }
    """
    # TODO: Implement the detailed parsing logic.
    # - Iterate through the trajectory.
    # - Identify messages from 'manager_agent' vs. 'search_agent'.
    # - For manager_agent, extract its tool calls (especially delegations
    #   to search_agent) as key "decisions".
    # - For search_agent, create a sub-trajectory or summary.
    # - Count total steps and steps that resulted in errors.
    # - Extract the final_answer from the last message.
    print("WARNING: `_parse_structured_trajectory` is not implemented. Returning dummy data.")
    
    # Placeholder implementation:
    manager_decisions = []
    agent_trajectory = {"manager_agent": [], "search_agent": []}
    final_answer = ""
    for step in trajectory:
        agent_name = step.get("name", "unknown")
        content = step.get("content", "")
        if "manager_agent" in agent_name:
            agent_trajectory["manager_agent"].append(content)
            # A simplistic way to guess a decision
            if "search_agent(" in content:
                manager_decisions.append(("delegate", "search_agent", content))
        elif "search_agent" in agent_name:
             agent_trajectory["search_agent"].append(content)
        
        if step.get("message_type") == "final_answer":
            final_answer = content

    return {
        "manager_decisions": manager_decisions,
        "final_answer": final_answer,
        "total_steps": len(trajectory),
        "error_steps": 0, # Placeholder
        "agent_trajectory": agent_trajectory
    }


# ==============================================================================
# 2. Reward Calculation Components
# ==============================================================================

def _calculate_base_reward(final_answer: str, ground_truth: str) -> float:
    """
    Calculates the primary reward based on the correctness of the final answer.

    Returns:
        1.0 for a correct answer, 0.0 for an incorrect one.
    """
    is_correct = ground_truth in final_answer
    return 1.0 if is_correct else 0.0


def _calculate_process_rewards(
    parsed_trajectory: Dict[str, Any],
    memory_manager: Any,
    is_correct: bool
) -> Dict[str, float]:
    """
    Calculates a set of rewards based on the quality of the agent's process.
    This is where "Reward Shaping" happens.

    Args:
        parsed_trajectory: The structured output from _parse_structured_trajectory.
        memory_manager: The agent's memory system to check for innovation/repetition.
        is_correct: A boolean indicating if the base reward was positive.

    Returns:
        A dictionary of different reward components, e.g.,
        {
            "innovation": 0.8,      # Rewarded for novel correct solutions
            "repetition": -0.9,     # Penalized for repeating past mistakes
            "efficiency": -0.15,    # Penalized for too many or failed steps
            "delegation": 0.2       # Rewarded for correctly using specialist agents
        }
    """
    # TODO: Implement the detailed reward shaping logic.
    # All values should be normalized, typically between -1.0 and 1.0.
    
    # 1. Innovation/Repetition (requires memory_manager and semantic similarity)
    # - Create a "strategy digest" from parsed_trajectory["manager_decisions"].
    # - If is_correct:
    #   - Compare digest to successful memories.
    #   - reward_innovation = 1 - max_similarity (encourage new solutions).
    # - If not is_correct:
    #   - Compare digest to failed memories.
    #   - penalty_repetition = max_similarity (penalize repeating mistakes).
    reward_innovation = 0.0
    penalty_repetition = 0.0

    # 2. Efficiency Penalty
    # - Penalize based on total steps and especially error steps.
    penalty_efficiency = (
        parsed_trajectory["total_steps"] * 0.01 +
        parsed_trajectory["error_steps"] * 0.1 # Heavier penalty for errors
    )

    # 3. Delegation Reward (Specific to Multi-Agent setups)
    # - Reward the manager for correctly delegating tasks.
    reward_delegation = 0.0
    for decision_type, agent_name, _ in parsed_trajectory["manager_decisions"]:
        if decision_type == "delegate" and agent_name == "search_agent":
            reward_delegation += 0.1 # Give a small bonus for each correct delegation

    return {
        "innovation": reward_innovation,
        "repetition": -penalty_repetition,
        "efficiency": -penalty_efficiency,
        "delegation": reward_delegation,
    }


# ==============================================================================
# 3. Main VERL-Compatible Reward Function
# ==============================================================================

def gaia_reward_func_with_memory(
    data_source: str,
    solution_str: str, # Note: This is often the raw model output string
    ground_truth: str,
    extra_info: dict = None
):
    """
    VERL-compatible reward function that orchestrates the calculation of a
    composite reward based on both the final answer and the agent's process.
    """
    # --- 1. Unpack data from the Agent Loop ---
    trajectory = extra_info.get("trajectory", [])
    memory_manager = extra_info.get("memory_manager", None)
    
    if not trajectory or not memory_manager:
        # Fallback to basic reward if critical info is missing
        return 1.0 if ground_truth in solution_str else 0.0

    # --- 2. Parse the Trajectory ---
    # This is the key step to understand the agent's strategy
    parsed_trajectory = _parse_structured_trajectory(trajectory)
    final_answer = parsed_trajectory.get("final_answer", solution_str)

    # --- 3. Calculate Reward Components ---
    r_base = _calculate_base_reward(final_answer, ground_truth)
    
    process_rewards = _calculate_process_rewards(
        parsed_trajectory,
        memory_manager,
        is_correct=(r_base > 0.5)
    )

    # --- 4. Combine Rewards with Weights ---
    weights = {
        "innovation": 0.1,
        "repetition": 0.2,
        "efficiency": 0.1,
        "delegation": 0.3 # Weighting delegation heavily encourages manager-like behavior
    }

    total_reward = (
        r_base
        + weights["innovation"] * process_rewards.get("innovation", 0.0)
        + weights["repetition"] * process_rewards.get("repetition", 0.0)
        + weights["efficiency"] * process_rewards.get("efficiency", 0.0)
        + weights["delegation"] * process_rewards.get("delegation", 0.0)
    )
    
    # Ensure reward is within a reasonable range, e.g., [-1, 2]
    return max(-1.0, min(2.0, total_reward))