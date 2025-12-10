"""State space enumeration via graph traversal."""

from collections import deque

from shopping_simulator.state import InventoryState, initial_state
from shopping_simulator.transitions import TransitionModel


def enumerate_reachable_states(
    model: TransitionModel,
    threshold: float,
    max_states: int = 500_000,
    prob_cutoff: float = 1e-12,
) -> list[InventoryState]:
    """
    Find all states reachable under the given threshold policy.

    Uses breadth-first search from an initial state, following all
    transitions with probability above the cutoff.

    Args:
        model: Transition dynamics model
        threshold: Stockout probability threshold for ordering
        max_states: Maximum states to enumerate (safety limit)
        prob_cutoff: Ignore transitions with probability below this

    Returns:
        List of all reachable states
    """
    start = initial_state(
        model.codelife, model.lead_time, model.units_per_case, model.lambda_d
    )

    visited: set[InventoryState] = {start}
    queue: deque[InventoryState] = deque([start])

    while queue and len(visited) < max_states:
        state = queue.popleft()

        try:
            transitions = model.get_transitions(state, threshold)
        except Exception as e:
            print(f"Warning: failed to get transitions from {state}: {e}")
            continue

        for next_state, prob, _, _, _ in transitions:
            if prob > prob_cutoff and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    if len(visited) >= max_states:
        print(
            f"Warning: reached max_states limit ({max_states}). "
            f"Results may be incomplete."
        )

    return list(visited)


def enumerate_with_multiple_starts(
    model: TransitionModel, threshold: float, max_states: int = 500_000
) -> list[InventoryState]:
    """
    Enumerate states starting from multiple initial conditions.

    This helps ensure we find the full recurrent class even if
    a single starting point misses some states.
    """
    all_states: set[InventoryState] = set()

    # Try different starting inventories
    for multiplier in [0.5, 1.0, 2.0]:
        start_units = int(model.codelife * model.lambda_d * multiplier)
        start_units = max(model.units_per_case, start_units)

        slots = [0] * model.n_slots
        slots[model.codelife - 1] = start_units
        start = InventoryState(tuple(slots))

        visited = {start}
        queue = deque([start])

        while queue and len(all_states) + len(visited) < max_states:
            state = queue.popleft()

            for next_state, prob, _, _, _ in model.get_transitions(state, threshold):
                if (
                    prob > 1e-12
                    and next_state not in visited
                    and next_state not in all_states
                ):
                    visited.add(next_state)
                    queue.append(next_state)

        all_states.update(visited)

    return list(all_states)
