"""Command-line interface for the analytical solver."""

import argparse
import time

from perishable.optimizer import AnalyticalInventory


def main():
    parser = argparse.ArgumentParser(description="Analytical inventory optimisation")
    parser.add_argument(
        "--codelife", type=int, default=3, help="Days product is sellable (default: 3)"
    )
    parser.add_argument(
        "--sales", type=float, default=5.0, help="Mean daily sales (default: 5.0)"
    )
    parser.add_argument(
        "--case-size", type=int, default=4, help="Units per case (default: 4)"
    )
    parser.add_argument(
        "--lead-time", type=int, default=3, help="Order lead time in days (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Evaluate specific threshold (default: find optimal)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="Grid points for optimization (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    solver = AnalyticalInventory(
        codelife=args.codelife,
        unit_sales_per_day=args.sales,
        units_per_case=args.case_size,
        lead_time=args.lead_time,
    )

    start_time = time.time()

    if args.threshold is not None:
        # Evaluate specific threshold
        result = solver.evaluate_threshold(args.threshold, verbose=args.verbose)
        print(f"\nResults for threshold = {args.threshold:.4f}:")
        print(f"  Total loss:        {result.total_loss:.4f}")
        print(f"  Availability loss: {result.availability_loss:.4f}")
        print(f"  Waste loss:        {result.waste_loss:.4f}")
    else:
        # Find optimal
        opt = solver.find_optimal_threshold(
            n_points=args.n_points,
            verbose=args.verbose,
        )

        print(f"\nOptimal threshold: {opt.optimal_threshold:.4f}")
        print(f"  Total loss:        {opt.optimal_loss.total_loss:.4f}")
        print(f"  Availability loss: {opt.optimal_loss.availability_loss:.4f}")
        print(f"  Waste loss:        {opt.optimal_loss.waste_loss:.4f}")
        print(f"  Mean cases/day:    {opt.optimal_loss.expected_cases_per_day:.2f}")
        print(f"  Mean shelf units:  {opt.optimal_loss.mean_available_units:.1f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
