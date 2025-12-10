"""Loss computation from steady-state distribution."""

from dataclasses import dataclass

import numpy as np

from shopping_simulator.markov import MarkovChain


@dataclass
class SteadyStateLoss:
    """Results from steady-state loss computation."""

    # Primary metrics (matching simulation output)
    total_loss: float
    availability_loss: float  # Stockout rate
    waste_loss: float  # Waste as fraction of ordered units

    # Additional metrics
    expected_waste_per_day: float
    expected_stockout_rate: float
    expected_cases_per_day: float
    mean_available_units: float

    # Full distribution for analysis
    stationary_distribution: np.ndarray


def compute_loss(chain: MarkovChain) -> SteadyStateLoss:
    """
    Compute expected losses from the stationary distribution.

    Matches the loss normalisation from the original simulation:
        - availability_loss = stockout rate (fraction of days with stockout)
        - waste_loss = wasted units / (cases ordered * units per case)
        - total_loss = availability_loss + waste_loss
    """
    pi = chain.solve_steady_state()

    # Expected values under stationary distribution
    expected_waste = float(np.dot(pi, chain.waste_per_state))
    expected_stockout = float(np.dot(pi, chain.stockout_per_state))
    expected_cases = float(np.dot(pi, chain.cases_ordered_per_state))

    # Mean available units on shelf
    mean_available = 0.0
    for prob, state in zip(pi, chain.states):
        mean_available += prob * state.available_units(chain.model.codelife)

    # Normalise waste to match simulation
    # Simulation: waste_loss = wasted_units / (cases_ordered * units_per_case)
    total_units_ordered = expected_cases * chain.model.units_per_case

    if total_units_ordered > 0:
        waste_loss = expected_waste / total_units_ordered
    else:
        waste_loss = 0.0

    availability_loss = expected_stockout
    total_loss = availability_loss + waste_loss

    return SteadyStateLoss(
        total_loss=total_loss,
        availability_loss=availability_loss,
        waste_loss=waste_loss,
        expected_waste_per_day=expected_waste,
        expected_stockout_rate=expected_stockout,
        expected_cases_per_day=expected_cases,
        mean_available_units=mean_available,
        stationary_distribution=pi,
    )
