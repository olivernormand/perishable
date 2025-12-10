# perishable

Markov chain model for perishable inventory optimisation.

Given product characteristics (shelf life, demand rate, case size, lead time), this package computes optimal ordering thresholds by solving for the steady-state distribution of the inventory system. It finds the policy that minimises combined waste and stockout losses without Monte Carlo simulation.

## Installation

```bash
pip install perishable
```

Or with uv:
```bash
uv add perishable
```

## Basic Usage

The basic usage of the package is as follows:

```python
from perishable import AnalyticalInventory

solver = AnalyticalInventory(
    codelife=5,           # Days product is sellable on shelf
    unit_sales_per_day=8, # Mean daily demand (Poisson)
    units_per_case=6,     # Units per case ordered
    lead_time=2           # Days from order to arrival
)

result = solver.find_optimal_threshold()

print(f"Optimal threshold: {result.optimal_threshold:.4f}")
print(f"Total loss: {result.optimal_loss.total_loss:.4f}")
print(f"Availability loss: {result.optimal_loss.availability_loss:.4f}")
print(f"Waste loss: {result.optimal_loss.waste_loss:.4f}")
```

This calculates the minimum loss (the sum of waste and unavailability) of the product in the store using an analytical Markov chain approach, which gives exact results without Monte Carlo sampling.

For evaluating a specific threshold:

```python
from perishable import AnalyticalInventory

solver = AnalyticalInventory(
    codelife=3,
    unit_sales_per_day=5,
    units_per_case=4,
    lead_time=3
)

print("Threshold | Total Loss | Availability | Waste")
print("-" * 50)
for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
    loss = solver.evaluate_threshold(t)
    print(f"{t:9.2f} | {loss.total_loss:10.4f} | {loss.availability_loss:12.4f} | {loss.waste_loss:.4f}")
```

Output:
```
Threshold | Total Loss | Availability | Waste
--------------------------------------------------
     0.05 |     0.2313 |       0.2222 | 0.0091
     0.10 |     0.4095 |       0.4075 | 0.0020
     0.20 |     0.4095 |       0.4075 | 0.0020
     0.30 |     0.5221 |       0.5215 | 0.0007
     0.50 |     0.6620 |       0.6611 | 0.0009
```

### Inspecting the State Space

```python
from perishable import AnalyticalInventory
from perishable.enumeration import enumerate_reachable_states
from perishable.markov import MarkovChain
from perishable.loss import compute_loss
import numpy as np

solver = AnalyticalInventory(
    codelife=3,
    unit_sales_per_day=2,
    units_per_case=4,
    lead_time=2
)

threshold = 0.1
states = enumerate_reachable_states(solver.model, threshold)
chain = MarkovChain(states, solver.model, threshold)
loss = compute_loss(chain)

print(f"Number of reachable states: {len(states)}")
print(f"Transition matrix density: {chain.P.nnz / len(states)**2:.4f}")

# Show top states by probability
pi = loss.stationary_distribution
top_indices = np.argsort(pi)[-5:][::-1]

print("\nTop 5 most likely states:")
print("Probability | On-Shelf  | In-Transit")
print("-" * 45)
for idx in top_indices:
    state = states[idx]
    on_shelf = state.slots[:3]   # codelife=3
    in_transit = state.slots[3:] # lead_time=2
    print(f"{pi[idx]:11.4f} | {str(on_shelf):9s} | {str(in_transit)}")
```

Output:
```
Number of reachable states: 28
Transition matrix density: 0.1964

Top 5 most likely states:
Probability | On-Shelf  | In-Transit
---------------------------------------------
     0.2625 | (0, 4, 4) | (0, 0)
     0.1853 | (4, 0, 4) | (4, 0)
     0.0828 | (2, 4, 0) | (4, 0)
     0.0756 | (3, 4, 0) | (4, 0)
     0.0633 | (1, 4, 0) | (4, 0)
```

### Analysing Lead Time Impact

```python
from perishable import AnalyticalInventory

print("Lead Time | Optimal Threshold | Total Loss | Availability | Waste")
print("-" * 70)

for lt in [1, 2, 3, 4, 5]:
    solver = AnalyticalInventory(
        codelife=5,
        unit_sales_per_day=3,
        units_per_case=6,
        lead_time=lt
    )
    result = solver.find_optimal_threshold(n_points=15)
    r = result.optimal_loss
    print(f"{lt:9d} | {result.optimal_threshold:17.4f} | {r.total_loss:10.4f} | {r.availability_loss:12.4f} | {r.waste_loss:.4f}")
```

Output:
```
Lead Time | Optimal Threshold | Total Loss | Availability | Waste
----------------------------------------------------------------------
        1 |            0.0800 |     0.0464 |       0.0212 | 0.0252
        2 |            0.0411 |     0.0605 |       0.0172 | 0.0433
        3 |            0.0567 |     0.0733 |       0.0398 | 0.0335
        4 |            0.0411 |     0.0812 |       0.0249 | 0.0563
        5 |            0.0411 |     0.0854 |       0.0315 | 0.0538
```

### Transition Analysis

Inspect what happens from a specific inventory state:

```python
from perishable import TransitionModel, InventoryState

model = TransitionModel(
    codelife=3,
    lead_time=2,
    units_per_case=4,
    unit_sales_per_day=2.0
)

# State: 4 fresh units on shelf, nothing in transit
# Slots: (expiring_today, 2_days_left, 3_days_left, arriving_tomorrow, arriving_in_2_days)
state = InventoryState((0, 0, 4, 0, 0))

print(f"Starting state: {state.slots}")
print(f"  On-shelf: {state.slots[:3]} (days 1, 2, 3 remaining)")
print(f"  In-transit: {state.slots[3:]} (arriving in 1, 2 days)")

# Get possible transitions at threshold 0.1
transitions = model.get_transitions(state, threshold=0.1)

print(f"\nPossible outcomes ({len(transitions)} states):")
print("Probability | Next State      | Stockout Prob")
print("-" * 50)
for next_state, prob, waste, stockout, cases in sorted(transitions, key=lambda x: -x[1]):
    print(f"{prob:11.4f} | {str(next_state.slots):15s} | {stockout:.4f}")
```

Output:
```
Starting state: (0, 0, 4, 0, 0)
  On-shelf: (0, 0, 4) (days 1, 2, 3 remaining)
  In-transit: (0, 0) (arriving in 1, 2 days)

Possible outcomes (5 states):
Probability | Next State      | Stockout Prob
--------------------------------------------------
     0.2707 | (0, 3, 0, 4, 0) | 0.0000
     0.2707 | (0, 2, 0, 4, 0) | 0.0000
     0.1804 | (0, 1, 0, 4, 0) | 0.0000
     0.1429 | (0, 0, 0, 4, 0) | 0.3685
     0.1353 | (0, 4, 0, 4, 0) | 0.0000
```

## API Reference

### AnalyticalInventory

Main interface for inventory optimisation.

```python
solver = AnalyticalInventory(
    codelife: int,              # Shelf life in days
    unit_sales_per_day: float,  # Mean daily demand
    units_per_case: int,        # Units per order
    lead_time: int,             # Order-to-arrival days
)

# Find optimal threshold
result = solver.find_optimal_threshold(
    n_points: int = 20,    # Grid search resolution
    refine: bool = True,   # Refine around minimum
    verbose: bool = False  # Print progress
)

# Evaluate specific threshold
loss = solver.evaluate_threshold(threshold: float)
```

### SteadyStateLoss

Results from loss computation:

- `total_loss`: Combined loss (availability + waste)
- `availability_loss`: Stockout rate
- `waste_loss`: Waste as fraction of ordered units
- `expected_cases_per_day`: Mean ordering rate
- `mean_available_units`: Mean stock on shelf

### InventoryState

Represents inventory configuration:

```python
state = InventoryState(slots=(0, 2, 4, 0, 6))
# slots[0:codelife] = on-shelf units by days remaining
# slots[codelife:] = in-transit units by days to arrival
```

### TransitionModel

Models daily inventory dynamics:

```python
model = TransitionModel(codelife, lead_time, units_per_case, unit_sales_per_day)
transitions = model.get_transitions(state, threshold)
```

## Architecture

```
perishable/
    __init__.py       # Package exports
    cli.py            # Command-line interface
    state.py          # InventoryState dataclass
    transitions.py    # Daily transition dynamics
    enumeration.py    # State space enumeration
    markov.py         # Markov chain construction
    loss.py           # Steady-state loss computation
    optimizer.py      # High-level optimisation API
```

## License

MIT
