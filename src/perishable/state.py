"""State representation for inventory systems."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryState:
    """
    Represents inventory as units at each days-remaining level.

    The slots tuple represents units at each freshness level:
        slots[0] = units with 1 day remaining (expire at end of today)
        slots[1] = units with 2 days remaining
        ...
        slots[codelife-1] = units with codelife days (freshest on shelf)
        slots[codelife] = units arriving tomorrow (1 day in transit)
        ...
        slots[codelife+lead_time-1] = units just ordered (lead_time days in transit)

    frozen=True makes instances hashable for use in sets and dict keys.
    """

    slots: tuple[int, ...]

    @property
    def total_units(self) -> int:
        """Total units across all slots."""
        return sum(self.slots)

    def available_units(self, codelife: int) -> int:
        """Units currently on shelf (sellable today)."""
        return sum(self.slots[:codelife])

    def in_transit(self, codelife: int) -> int:
        """Units ordered but not yet on shelf."""
        return sum(self.slots[codelife:])

    def to_simulation_format(self, codelife: int) -> tuple[list[int], list[int]]:
        """
        Convert to (n_units, days_left) format used by the original simulation.

        Returns:
            n_units: list of unit counts for each batch
            days_left: list of days remaining for each batch
        """
        n_units = []
        days_left = []

        for i, units in enumerate(self.slots):
            if units > 0:
                n_units.append(units)
                days_left.append(i + 1)  # slots[i] has i+1 days remaining

        return n_units, days_left

    @classmethod
    def from_simulation_format(
        cls, n_units: list[int], days_left: list[int], n_slots: int
    ) -> "InventoryState":
        """
        Create state from (n_units, days_left) format.

        Args:
            n_units: list of unit counts for each batch
            days_left: list of days remaining for each batch
            n_slots: total number of slots (codelife + lead_time)
        """
        slots = [0] * n_slots

        for units, days in zip(n_units, days_left):
            if 1 <= days <= n_slots:
                slots[days - 1] += units

        return cls(tuple(slots))


def initial_state(
    codelife: int, lead_time: int, units_per_case: int, unit_sales_per_day: float
) -> InventoryState:
    """
    Create a reasonable starting state for enumeration.

    Starts with approximately enough stock to cover demand over the codelife,
    placed as fresh stock (maximising shelf life).
    """
    n_slots = codelife + lead_time
    slots = [0] * n_slots

    # Start with enough stock to roughly meet demand, rounded to case multiples
    target_units = int(codelife * unit_sales_per_day)
    n_cases = max(1, (target_units + units_per_case - 1) // units_per_case)

    # Place as fresh stock on shelf
    slots[codelife - 1] = n_cases * units_per_case

    return InventoryState(tuple(slots))
