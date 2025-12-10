"""Transition dynamics for the inventory system."""

from functools import lru_cache

import numpy as np
from scipy.stats import poisson

from shopping_simulator.state import InventoryState


class TransitionModel:
    """
    Models single-day transitions in the inventory system.

    Each day follows this sequence:
        1. Check P(stockout) and order cases until below threshold
        2. Demand realises (Poisson distributed)
        3. Sales occur with rotation (oldest stock sold first)
        4. Unsold stock with 1 day remaining becomes waste
        5. All stock ages by one day
    """

    def __init__(
        self,
        codelife: int,
        lead_time: int,
        units_per_case: int,
        unit_sales_per_day: float,
        max_units_per_slot: int | None = None,
    ):
        """
        Args:
            codelife: Days a product is sellable once on shelf
            lead_time: Days between ordering and arrival
            units_per_case: Units in each case ordered
            unit_sales_per_day: Mean daily demand (Poisson parameter)
            max_units_per_slot: Cap on units per slot (for state space bounds)
        """
        self.codelife = codelife
        self.lead_time = lead_time
        self.units_per_case = units_per_case
        self.lambda_d = unit_sales_per_day
        self.n_slots = codelife + lead_time

        # State space bounds
        if max_units_per_slot is None:
            # Default: enough to cover ~10 standard deviations of demand over codelife
            max_demand = unit_sales_per_day * codelife + 10 * np.sqrt(
                unit_sales_per_day * codelife
            )
            self.max_units_per_slot = int(max_demand * 2)
        else:
            self.max_units_per_slot = max_units_per_slot

        # Precompute demand distribution
        # Cover up to ~6 standard deviations above mean
        self.max_demand = int(self.lambda_d + 6 * np.sqrt(self.lambda_d)) + 10
        self.demand_probs = poisson.pmf(np.arange(self.max_demand + 1), self.lambda_d)
        # Ensure probabilities sum to 1 (lump tail into max)
        tail_prob = 1 - self.demand_probs.sum()
        if tail_prob > 0:
            self.demand_probs[-1] += tail_prob

    def apply_ordering(self, state: InventoryState, threshold: float) -> InventoryState:
        """
        Apply ordering policy: add cases until P(stockout) <= threshold.

        Returns new state after ordering decisions.
        """
        slots = list(state.slots)

        # Check if we need to order
        p_stockout = self.probability_stockout(state)

        while p_stockout > threshold:
            # Order one case (arrives in lead_time days)
            slots[-1] += self.units_per_case

            # Cap to prevent unbounded state space
            if slots[-1] > self.max_units_per_slot:
                slots[-1] = self.max_units_per_slot
                break

            new_state = InventoryState(tuple(slots))
            p_stockout = self.probability_stockout(new_state)

        return InventoryState(tuple(slots))

    def count_cases_ordered(
        self, state_before: InventoryState, state_after: InventoryState
    ) -> int:
        """Count how many cases were ordered between two states."""
        diff = state_after.slots[-1] - state_before.slots[-1]
        return max(0, diff // self.units_per_case)

    def probability_stockout(self, state: InventoryState) -> float:
        """
        Calculate probability of stockout at lead_time+1 days ahead.

        Uses the rotation model: oldest stock is sold first, and we need
        to track probability distributions through time intervals.
        """
        # Convert to hashable format for caching
        return self._probability_stockout_cached(state.slots)

    @lru_cache(maxsize=100000)
    def _probability_stockout_cached(self, slots: tuple[int, ...]) -> float:
        """Cached implementation of stockout probability calculation."""

        n_units = np.array(slots)
        days_left = np.arange(1, len(slots) + 1)

        max_days_ahead = self.lead_time + 1

        # Check for active stock at end of the lookahead period
        # After max_days_ahead-1 days of aging, stock with days_left=d will have d-(max_days_ahead-1) days left
        # It's active if: still has time (days_left > max_days_ahead-1) AND on shelf (days_left - (max_days_ahead-1) <= codelife)
        remaining_days = days_left - (max_days_ahead - 1)
        active_at_end = (remaining_days > 0) & (remaining_days <= self.codelife)

        if not np.any(active_at_end & (n_units > 0)):
            return 1.0

        # Build intervals where stock composition changes
        # Changes happen when: stock expires (days_left passes codelife) or arrives (days_left passes codelife from above)
        interval_edges = [0, max_days_ahead]

        for d in days_left:
            if 0 < d <= max_days_ahead:
                interval_edges.append(d)
            if 0 < d - self.codelife <= max_days_ahead:
                interval_edges.append(d - self.codelife)

        interval_edges = sorted(set(interval_edges))
        interval_edges = [e for e in interval_edges if e <= max_days_ahead]
        intervals = np.diff(interval_edges)

        # Initialize probability distributions for each slot
        # p_units_list[i] is probability distribution over units remaining in slot i
        p_units_list = []
        for units in n_units:
            p = np.zeros(units + 1)
            p[-1] = 1.0  # Start with certainty of having 'units' units
            p_units_list.append(p)

        # Process each interval
        days_ahead = 0
        active_idx = None

        for interval in intervals:
            # Determine which slots are active during this interval
            # Active = not expired AND on shelf
            remaining = days_left - days_ahead
            not_expired = remaining > 0
            on_shelf = remaining <= self.codelife
            active = not_expired & on_shelf & (n_units > 0)
            active_idx = np.where(active)[0]

            if len(active_idx) == 0:
                days_ahead += interval
                continue

            # Expected sales in this interval
            expected_sales = self.lambda_d * interval
            max_sales = int(expected_sales + 10 * np.sqrt(max(expected_sales, 1))) + 1

            # Probability distribution over sales
            p_sales = poisson.pmf(np.arange(max_sales), expected_sales)
            p_sales = p_sales / p_sales.sum()

            # Propagate probability through active slots (oldest first = rotation)
            active_p_units = [p_units_list[i] for i in active_idx]
            active_p_units = self._propagate_probability(p_sales, active_p_units)

            # Update the distributions
            for j, i in enumerate(active_idx):
                p_units_list[i] = active_p_units[j]

            days_ahead += interval

        # Probability of stockout = probability all active slots have 0 units
        if active_idx is None or len(active_idx) == 0:
            return 1.0

        prob_stockout = 1.0
        for i in active_idx:
            prob_stockout *= p_units_list[i][0]

        return float(prob_stockout)

    def _propagate_probability(
        self, p_sales: np.ndarray, p_units_list: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Propagate sales probability through inventory slots (rotation order).

        Sales are applied to oldest stock first. Remaining demand passes to next slot.
        """
        updated_list = []

        for p_units in p_units_list:
            p_sales, p_units_updated = self._update_units_sales(p_sales, p_units)
            updated_list.append(p_units_updated)

        return updated_list

    def _update_units_sales(
        self, p_sales: np.ndarray, p_units: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given demand and stock distributions, compute remaining demand and stock.

        Returns:
            p_sales_remaining: distribution of unfulfilled demand
            p_units_remaining: distribution of remaining stock
        """
        n_sales = len(p_sales)
        n_units = len(p_units)

        # Joint probability matrix: p_matrix[s, u] = P(sales=s) * P(units=u)
        p_matrix = np.outer(p_sales, p_units)

        # After selling min(s, u) units:
        # - Remaining sales = max(s - u, 0)
        # - Remaining units = max(u - s, 0)

        p_sales_remaining = np.zeros(n_sales)
        p_units_remaining = np.zeros(n_units)

        # Units remaining: sum along diagonals where units > sales
        for u in range(n_units):
            # P(remaining = u) = sum over s where original_units - s = u, i.e., s = original_units - u
            # This is the diagonal offset by u
            p_units_remaining[u] = np.trace(p_matrix, offset=u)

        # Sales remaining: sum along diagonals where sales > units
        for s in range(n_sales):
            p_sales_remaining[s] = np.trace(p_matrix, offset=-s)

        # Normalise to handle numerical issues
        p_units_sum = p_units_remaining.sum()
        if p_units_sum > 0:
            p_units_remaining /= p_units_sum
        else:
            p_units_remaining[0] = 1.0

        p_sales_sum = p_sales_remaining.sum()
        if p_sales_sum > 0:
            p_sales_remaining /= p_sales_sum
        else:
            p_sales_remaining[0] = 1.0

        return p_sales_remaining, p_units_remaining

    def apply_demand_and_age(
        self, state: InventoryState, demand: int
    ) -> tuple[InventoryState, int, bool]:
        """
        Apply a specific demand realisation, then age inventory.

        Args:
            state: Current inventory state
            demand: Number of units demanded

        Returns:
            new_state: State after sales and aging
            waste: Units that expired
            stockout: Whether demand exceeded supply
        """
        slots = list(state.slots)
        remaining_demand = demand

        # Sell with rotation (oldest first)
        for i in range(self.codelife):
            sold = min(slots[i], remaining_demand)
            slots[i] -= sold
            remaining_demand -= sold

        # Stockout if we couldn't fulfill all demand
        stockout = remaining_demand > 0

        # Waste = units in slot 0 (1 day remaining) that weren't sold
        waste = slots[0]

        # Age: shift everything down by one slot
        aged_slots = slots[1:] + [0]

        return InventoryState(tuple(aged_slots)), waste, stockout

    def get_transitions(
        self, state: InventoryState, threshold: float
    ) -> list[tuple[InventoryState, float, int, bool, int]]:
        """
        Get all possible transitions from a state under given threshold policy.

        Returns list of (next_state, probability, waste, stockout, cases_ordered)
        """
        # Apply ordering (deterministic given state and threshold)
        state_after_order = self.apply_ordering(state, threshold)
        cases_ordered = self.count_cases_ordered(state, state_after_order)

        # Aggregate transitions by next_state (different demands may lead to same state)
        # We need to track probability-weighted waste and stockout since different
        # demand levels can lead to the same next_state but with different outcomes
        transitions_dict: dict[InventoryState, tuple[float, float, float]] = {}

        for demand in range(self.max_demand + 1):
            next_state, waste, stockout = self.apply_demand_and_age(
                state_after_order, demand
            )
            prob = self.demand_probs[demand]

            if next_state in transitions_dict:
                old_prob, old_waste_weighted, old_stockout_weighted = transitions_dict[
                    next_state
                ]
                # Accumulate probability-weighted waste and stockout
                transitions_dict[next_state] = (
                    old_prob + prob,
                    old_waste_weighted + prob * waste,
                    old_stockout_weighted + prob * float(stockout),
                )
            else:
                transitions_dict[next_state] = (
                    prob,
                    prob * waste,
                    prob * float(stockout),
                )

        # Convert weighted sums to conditional expectations given next_state
        return [
            (
                next_state,
                prob,
                waste_weighted / prob if prob > 0 else 0.0,
                stockout_weighted / prob if prob > 0 else 0.0,
                cases_ordered,
            )
            for next_state, (
                prob,
                waste_weighted,
                stockout_weighted,
            ) in transitions_dict.items()
        ]
