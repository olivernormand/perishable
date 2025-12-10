"""Markov chain construction and steady-state solution."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from shopping_simulator.state import InventoryState
from shopping_simulator.transitions import TransitionModel


class MarkovChain:
    """
    Sparse Markov chain for inventory dynamics.

    Constructs the transition matrix P where P[i,j] is the probability
    of transitioning from state i to state j in one day.
    """

    def __init__(
        self,
        states: list[InventoryState],
        model: TransitionModel,
        threshold: float,
        verbose: bool = False,
    ):
        """
        Build the Markov chain for the given states and policy.

        Args:
            states: List of all states in the chain
            model: Transition dynamics model
            threshold: Ordering threshold policy
            verbose: Print progress during construction
        """
        self.states = states
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.n_states = len(states)
        self.model = model
        self.threshold = threshold

        # Per-state expected values (computed during matrix construction)
        self.waste_per_state = np.zeros(self.n_states)
        self.stockout_per_state = np.zeros(self.n_states)
        self.cases_ordered_per_state = np.zeros(self.n_states)

        # Build the transition matrix
        self.P: csr_matrix = self._build_matrix(verbose)

    def _build_matrix(self, verbose: bool) -> csr_matrix:
        """Construct sparse transition matrix."""
        P = lil_matrix((self.n_states, self.n_states))

        for i, state in enumerate(self.states):
            if verbose and i % 1000 == 0:
                print(f"  Processing state {i}/{self.n_states}")

            transitions = self.model.get_transitions(state, self.threshold)

            expected_waste = 0.0
            expected_stockout = 0.0
            expected_cases = 0.0
            total_prob = 0.0

            for next_state, prob, waste, stockout, cases_ordered in transitions:
                if next_state in self.state_to_idx:
                    j = self.state_to_idx[next_state]
                    P[i, j] += prob
                    total_prob += prob

                expected_waste += prob * waste
                expected_stockout += prob * float(stockout)
                expected_cases += prob * cases_ordered

            self.waste_per_state[i] = expected_waste
            self.stockout_per_state[i] = expected_stockout
            self.cases_ordered_per_state[i] = expected_cases

            # Normalise row if needed (handles probability mass to unreachable states)
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-10:
                P[i, :] /= total_prob

        return P.tocsr()

    def solve_steady_state(
        self, method: str = "power", tol: float = 1e-12, max_iter: int = 100_000
    ) -> np.ndarray:
        """
        Find the stationary distribution π where πP = π.

        Args:
            method: 'power' for power iteration, 'eigen' for eigenvector method
            tol: Convergence tolerance
            max_iter: Maximum iterations for power method

        Returns:
            Stationary distribution as numpy array
        """
        if method == "power":
            return self._solve_power_iteration(tol, max_iter)
        elif method == "eigen":
            return self._solve_eigenvector()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_power_iteration(self, tol: float, max_iter: int) -> np.ndarray:
        """Solve using power iteration with fallback for periodic chains."""
        # Start with uniform distribution
        pi = np.ones(self.n_states) / self.n_states

        for iteration in range(max_iter):
            pi_new = pi @ self.P

            # Renormalise for numerical stability
            pi_sum = pi_new.sum()
            if pi_sum > 0:
                pi_new /= pi_sum

            # Check convergence
            diff = np.abs(pi_new - pi).max()
            if diff < tol:
                return pi_new

            pi = pi_new

        # Power iteration didn't converge - likely a periodic chain
        # Fall back to eigenvector method which handles periodicity correctly
        if diff > 1e-6:
            # Oscillation indicates periodicity or slow convergence, use eigenvector method
            return self._solve_eigenvector()

        # Essentially converged but didn't meet strict tolerance - return current estimate
        return pi

    def _solve_eigenvector(self) -> np.ndarray:
        """Solve for the time-averaged stationary distribution.

        For periodic chains, we use Cesàro averaging: compute the average
        distribution over many steps, which converges to the unique time-averaged
        stationary distribution regardless of periodicity.
        """
        # Use Cesàro averaging: average pi over many iterations
        # This converges for both periodic and aperiodic chains
        pi = np.ones(self.n_states) / self.n_states
        pi_sum = pi.copy()

        n_iterations = 1000
        for i in range(1, n_iterations):
            pi = pi @ self.P
            pi_sum += pi

        pi_avg = pi_sum / n_iterations
        pi_avg /= pi_avg.sum()

        return pi_avg

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the chain."""
        return {
            "n_states": self.n_states,
            "nnz": self.P.nnz,
            "density": self.P.nnz / (self.n_states**2),
            "avg_transitions_per_state": self.P.nnz / self.n_states,
        }
