# From Monte Carlo to Markov Chains: Exact Inventory Optimisation

## Introduction

The shopping-simulator package determines optimal ordering policies for perishable products in retail. Given four parameters - shelf life (codelife), daily sales rate, case size, and order lead time - it finds the ordering threshold that minimises total loss from waste and stockouts.

The original implementation used Monte Carlo simulation: run thousands of simulated days, try different ordering thresholds, and find which performs best. This worked, but had limitations around runtime, statistical noise, and convergence.

We've replaced this with an **analytical Markov chain approach** that computes exact steady-state losses in seconds rather than minutes, with no sampling variance.

## The Problem

A retailer orders perishable products in cases. Each day:
1. Check current inventory and decide whether to order
2. Customer demand arrives (Poisson distributed)
3. Sales occur using FIFO rotation (oldest stock sold first)
4. Unsold stock that has reached its expiry date becomes waste
5. Remaining stock ages by one day

The ordering policy uses a **stockout probability threshold**: order more cases when the probability of running out before the next delivery exceeds this threshold.

The goal is to find the threshold that minimises:
```
total_loss = availability_loss + waste_loss
```

Where:
- `availability_loss` = fraction of days with stockouts
- `waste_loss` = wasted units / ordered units

## The Old Approach: Monte Carlo Simulation

The original `LossSimulation` class ran stochastic simulations:

```python
sim = LossSimulation(codelife=3, unit_sales_per_day=5, units_per_case=4, lead_time=3)
min_loss, std = sim.calculate_min_loss_and_variance(total_days=20000)
```

This would:
1. Simulate 20,000 days of inventory operations
2. Try 10-20 different threshold values
3. Optionally refine around the best threshold
4. Return the minimum loss found

**Problems:**
- **Slow**: 20,000 simulated days per threshold evaluation
- **Noisy**: Results varied between runs due to random sampling
- **Uncertain**: Hard to know if you've found the true optimum
- **Memory**: Needed to track simulation state over many days

## The New Approach: Markov Chain Steady State

The inventory system is a Markov chain. Each state represents the current inventory configuration:
- Units on shelf at each remaining freshness level
- Units in transit at each delivery distance

The key insight: we don't need to simulate - we can compute the **stationary distribution** directly and derive exact expected losses.

### State Representation

```python
@dataclass(frozen=True)
class InventoryState:
    slots: tuple[int, ...]  # (slot_0, slot_1, ..., slot_{codelife+lead_time-1})
```

Where:
- `slots[0]` = units expiring today (1 day of shelf life left)
- `slots[codelife-1]` = fresh units (full shelf life remaining)
- `slots[codelife:]` = units in transit (arriving in 1, 2, ... lead_time days)

### Building the Transition Matrix

For each state, we enumerate all possible transitions:

```python
def get_transitions(self, state, threshold):
    # 1. Apply ordering policy (deterministic given state and threshold)
    state_after_order = self.apply_ordering(state, threshold)
    
    # 2. For each possible demand realisation (Poisson)
    for demand in range(max_demand):
        next_state, waste, stockout = self.apply_demand_and_age(state_after_order, demand)
        prob = poisson_pmf(demand, mean_sales)
        # Record transition probability and associated waste/stockout
```

This gives us a sparse transition matrix P where P[i,j] is the probability of moving from state i to state j in one day.

### Computing Steady State

For an ergodic Markov chain, there exists a unique stationary distribution π where:
```
π P = π
```

We solve this using power iteration with Cesaro averaging to handle periodic chains:

```python
def _solve_eigenvector(self):
    pi = uniform_distribution()
    pi_sum = pi.copy()
    
    for i in range(1000):
        pi = pi @ P
        pi_sum += pi
    
    return pi_sum / 1000  # Time-averaged distribution
```

### Computing Expected Loss

Once we have π, the expected losses are simple dot products:

```python
expected_waste = np.dot(pi, waste_per_state)
expected_stockout = np.dot(pi, stockout_per_state)
```

Where `waste_per_state[i]` and `stockout_per_state[i]` are the expected waste and stockout probability when starting from state i (computed during transition matrix construction).

## Bugs Found and Fixed

### Bug 1: Incorrect Aggregation of Waste/Stockout

When building transitions, multiple demand levels can lead to the same next state. The original code kept only the first waste/stockout values:

```python
# BUG: Overwrites with first values seen
if next_state in transitions_dict:
    transitions_dict[next_state] = (old_prob + prob, old_waste, old_stockout)
```

This caused massive underreporting of stockouts. For example, with 2 units on shelf and mean demand 8:
- Demand 0-2: stockout=False (prob ~1.4%)
- Demand 3+: stockout=True (prob ~98.6%)

All demands led to the same empty-shelf next state, but the code reported 0% stockout.

**Fix**: Use probability-weighted aggregation:

```python
transitions_dict[next_state] = (
    old_prob + prob,
    old_waste_weighted + prob * waste,
    old_stockout_weighted + prob * float(stockout),
)
```

### Bug 2: Periodic Chain Handling

When `lead_time` creates cyclical ordering patterns (e.g., orders arriving every 3 days), the Markov chain becomes periodic with multiple eigenvalues equal to 1.

Standard power iteration oscillates forever on periodic chains. Our initial eigenvector fix used `scipy.sparse.linalg.eigs` to find the eigenvalue-1 eigenvector, but this only returned one eigenvector, giving an incorrect distribution.

**Fix**: Use Cesaro averaging - average the distribution over many iterations. This converges to the correct time-averaged stationary distribution regardless of periodicity:

```python
def _solve_eigenvector(self):
    pi = np.ones(n_states) / n_states
    pi_sum = pi.copy()
    
    for i in range(1000):
        pi = pi @ self.P
        pi_sum += pi
    
    return pi_sum / 1000
```

This fixed a critical bug where longer lead times appeared to have *lower* loss - clearly wrong since you have less ability to react to demand.

## Results Comparison

### Speed

| Method | Time for codelife=5, sales=8, case_size=6, lead_time=2 |
|--------|--------------------------------------------------------|
| Monte Carlo (20k days) | ~30 seconds |
| Analytical | ~7 seconds |

### Accuracy

Monte Carlo results vary between runs. Analytical results are exact (to numerical precision).

### Example Output

```
$ python -m shopping_simulator.cli --codelife 5 --sales 8 --case-size 6 --lead-time 2

Optimal threshold: 0.0100
  Total loss:        0.0875
  Availability loss: 0.0875
  Waste loss:        0.0000
  Mean cases/day:    1.29
  Mean shelf units:  14.2

Completed in 7.11 seconds
```

## Architecture

The new implementation consists of:

```
shopping_simulator/
    __init__.py          # Package exports
    cli.py               # Command-line interface
    state.py             # InventoryState dataclass
    transitions.py       # TransitionModel - daily dynamics
    enumeration.py       # State space enumeration
    markov.py            # MarkovChain - matrix construction and solving
    loss.py              # SteadyStateLoss computation
    optimizer.py         # AnalyticalInventory - high-level API
```

### Usage

```python
from shopping_simulator import AnalyticalInventory

solver = AnalyticalInventory(
    codelife=5,
    unit_sales_per_day=8,
    units_per_case=6,
    lead_time=2
)

# Find optimal threshold
result = solver.find_optimal_threshold()
print(f"Optimal threshold: {result.optimal_threshold}")
print(f"Total loss: {result.optimal_loss.total_loss}")

# Evaluate specific threshold
loss = solver.evaluate_threshold(0.1)
```

## Conclusion

The Markov chain approach provides exact solutions where Monte Carlo gave noisy estimates. It's faster, deterministic, and mathematically grounded.

The key insight is that inventory dynamics form a finite-state Markov chain. Once you enumerate the reachable states and build the transition matrix, computing long-run averages becomes a linear algebra problem rather than a simulation problem.

This transformation from simulation to analysis is a common pattern in operations research - whenever you have a stochastic process with finite state space, consider whether you can compute steady-state properties directly rather than simulating.
