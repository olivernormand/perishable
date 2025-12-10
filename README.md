## Shopping Simulator

This package determines the best availability and waste characteristics of a product in a store. This depends on four parameters:

1. `codelife` defines the number of days a product is available in the store once it hits the shelves.
2. `unit_sales_per_day` defines the number of units sold per day.
3. `units_per_case` defines the number of units in a case. The case is the fundamental unit of the product in the retail supply chain.
4. `lead_time` defines the number of days taken for the demand signal to lead to a stock replenishment.

Based on these parameters, `shopping-simulator` is able to determine the best availability and waste performance of a product in the store. 

---
## Basic Usage

The basic usage of the package is as follows:

```python
from shopping_simulator.optimizer import AnalyticalInventory

solver = AnalyticalInventory(
    codelife=3, 
    unit_sales_per_day=5, 
    units_per_case=4, 
    lead_time=3
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
loss = solver.evaluate_threshold(0.1)
print(f"Loss at threshold 0.1: {loss.total_loss:.4f}")
```
