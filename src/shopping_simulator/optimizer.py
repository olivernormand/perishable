"""High-level interface for analytical inventory optimisation."""

from dataclasses import dataclass

import numpy as np

from shopping_simulator.enumeration import enumerate_reachable_states
from shopping_simulator.loss import SteadyStateLoss, compute_loss
from shopping_simulator.markov import MarkovChain
from shopping_simulator.transitions import TransitionModel


@dataclass
class OptimizationResult:
    """Results from threshold optimization."""

    optimal_threshold: float
    optimal_loss: SteadyStateLoss
    all_thresholds: np.ndarray
    all_losses: list[SteadyStateLoss]


class AnalyticalInventory:
    """
    Analytical solver for inventory optimisation.

    Replaces Monte Carlo simulation with exact Markov chain steady-state
    computation for finding optimal ordering policies.
    """

    def __init__(
        self,
        codelife: int,
        unit_sales_per_day: float,
        units_per_case: int,
        lead_time: int,
        max_units_per_slot: int | None = None,
    ):
        """
        Args:
            codelife: Days product is sellable on shelf
            unit_sales_per_day: Mean daily demand
            units_per_case: Units per case ordered
            lead_time: Days from order to arrival
            max_units_per_slot: Optional cap on units per freshness level
        """
        self.model = TransitionModel(
            codelife=codelife,
            lead_time=lead_time,
            units_per_case=units_per_case,
            unit_sales_per_day=unit_sales_per_day,
            max_units_per_slot=max_units_per_slot,
        )

        # Cache for state enumerations
        self._state_cache: dict[float, list] = {}

    def evaluate_threshold(
        self, threshold: float, verbose: bool = False
    ) -> SteadyStateLoss:
        """
        Compute exact expected loss for a given ordering threshold.

        Args:
            threshold: P(stockout) threshold - order when above this
            verbose: Print progress information

        Returns:
            Steady-state loss metrics
        """
        if verbose:
            print(f"Evaluating threshold {threshold:.4f}")

        # Enumerate reachable states
        if verbose:
            print("  Enumerating states...")
        states = enumerate_reachable_states(self.model, threshold)

        if verbose:
            print(f"  Found {len(states)} reachable states")
            print("  Building transition matrix...")

        # Build Markov chain
        chain = MarkovChain(states, self.model, threshold, verbose=verbose)

        if verbose:
            diag = chain.get_diagnostics()
            print(
                f"  Matrix: {diag['n_states']} states, {diag['nnz']} non-zeros, "
                f"density {diag['density']:.2e}"
            )
            print("  Solving for steady state...")

        # Compute loss
        result = compute_loss(chain)

        if verbose:
            print(
                f"  Loss: {result.total_loss:.4f} "
                f"(avail: {result.availability_loss:.4f}, waste: {result.waste_loss:.4f})"
            )

        return result

    def find_optimal_threshold(
        self,
        thresholds: np.ndarray | None = None,
        n_points: int = 20,
        refine: bool = True,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Find the threshold that minimises total loss.

        Args:
            thresholds: Explicit thresholds to evaluate, or None for automatic
            n_points: Number of points for initial grid (if thresholds is None)
            refine: Whether to refine around the minimum
            verbose: Print progress

        Returns:
            OptimizationResult with optimal threshold and all evaluated points
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, n_points)

        if verbose:
            print(f"Searching {len(thresholds)} threshold values...")

        all_losses = []
        for threshold in thresholds:
            result = self.evaluate_threshold(threshold, verbose=verbose)
            all_losses.append(result)

        # Find minimum
        losses = np.array([r.total_loss for r in all_losses])
        best_idx = np.argmin(losses)

        # Optionally refine around minimum
        if refine and len(thresholds) >= 3:
            if verbose:
                print("Refining around minimum...")

            # Search window around best point
            lower_idx = max(0, best_idx - 1)
            upper_idx = min(len(thresholds) - 1, best_idx + 1)

            refined_thresholds = np.linspace(
                thresholds[lower_idx], thresholds[upper_idx], 10
            )

            for threshold in refined_thresholds:
                if threshold not in thresholds:
                    result = self.evaluate_threshold(threshold, verbose=verbose)
                    all_losses.append(result)
                    thresholds = np.append(thresholds, threshold)

            # Re-sort and find new minimum
            sort_idx = np.argsort(thresholds)
            thresholds = thresholds[sort_idx]
            all_losses = [all_losses[i] for i in sort_idx]

            losses = np.array([r.total_loss for r in all_losses])
            best_idx = np.argmin(losses)

        return OptimizationResult(
            optimal_threshold=float(thresholds[best_idx]),
            optimal_loss=all_losses[best_idx],
            all_thresholds=thresholds,
            all_losses=all_losses,
        )

    def compare_with_simulation(
        self, threshold: float, simulation_days: int = 10000
    ) -> dict:
        """
        Compare analytical results with Monte Carlo simulation.

        Useful for validation.
        """
        # Analytical result
        analytical = self.evaluate_threshold(threshold)

        # Run equivalent simulation (import your existing simulator)
        from shopping_simulator.simulator import LossSimulation

        sim = LossSimulation(
            codelife=self.model.codelife,
            unit_sales_per_day=int(self.model.lambda_d),
            units_per_case=self.model.units_per_case,
            lead_time=self.model.lead_time,
        )

        sim_loss, sim_avail, sim_waste, _, _ = sim.calculate_loss(
            total_days=simulation_days,
            stockout_threshold=threshold,
            rotation=True,
        )

        return {
            "threshold": threshold,
            "analytical": {
                "total_loss": analytical.total_loss,
                "availability_loss": analytical.availability_loss,
                "waste_loss": analytical.waste_loss,
            },
            "simulation": {
                "total_loss": sim_loss,
                "availability_loss": sim_avail,
                "waste_loss": sim_waste,
            },
            "difference": {
                "total_loss": analytical.total_loss - sim_loss,
                "availability_loss": analytical.availability_loss - sim_avail,
                "waste_loss": analytical.waste_loss - sim_waste,
            },
        }
