"""Command-line entry point for pricing experiments and runtime diagnostics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .environment import EnvironmentReport

if TYPE_CHECKING:
    from .black_scholes import BlackScholesResult
    from .problems import PricingCase


def _monte_carlo_summary(
    args: argparse.Namespace, case: "PricingCase", reference_value: "BlackScholesResult",
) -> dict[str, object]:
    import numpy as np

    from .monte_carlo import MonteCarloPricer

    rng = np.random.default_rng(args.seed)
    pricer = MonteCarloPricer(case.model, case.contract)
    summary: dict[str, object] = {
        "case": case.name, "seed": args.seed,
        "bit_generator": type(rng.bit_generator).__name__,
    }
    if args.quantity == "price":
        price_result = pricer.estimate(case.initial_spots, n_paths=args.paths, rng=rng)
        error = price_result.price_estimate - reference_value.call_price
        summary.update({
            "method": "exact_terminal_monte_carlo", **asdict(price_result),
            "reference_call_price": reference_value.call_price,
            "signed_error": error,
            "error_in_standard_errors": (
                error / price_result.standard_error
                if price_result.standard_error > 0 else None
            ),
        })
    else:
        reference_delta = reference_value.call_delta
        if reference_delta is None:
            raise ValueError("The case has no defined analytical spot delta.")
        delta_result = pricer.estimate_delta(
            case.initial_spots, n_paths=args.paths, rng=rng,
            bump_fraction=args.bump_fraction,
        )
        error = delta_result.pathwise.estimate - reference_delta
        summary.update({
            "method": "pathwise_and_common_random_number_delta", **asdict(delta_result),
            "reference_call_delta": reference_delta,
            "pathwise_signed_error": error,
            "pathwise_error_in_standard_errors": (
                error / delta_result.pathwise.standard_error
                if delta_result.pathwise.standard_error > 0 else None
            ),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bsde_pricing")
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser(
        "environment", help="check numerical dependencies and report the runtime"
    )
    describe = commands.add_parser(
        "describe", help="show a pricing case and its financial conventions"
    )
    describe.add_argument("--case", choices=("vanilla",), default="vanilla")
    reference = commands.add_parser(
        "reference", help="evaluate the analytical price and spot Greeks"
    )
    reference.add_argument("--case", choices=("vanilla",), default="vanilla")
    monte_carlo = commands.add_parser(
        "monte-carlo", help="estimate a price and its uncertainty from IID terminal states"
    )
    monte_carlo.add_argument("--case", choices=("vanilla",), default="vanilla")
    monte_carlo.add_argument("--paths", type=int, default=100_000)
    monte_carlo.add_argument("--seed", type=int, default=2026)
    monte_carlo.add_argument("--quantity", choices=("price", "delta"), default="price")
    monte_carlo.add_argument(
        "--bump-fraction", type=float, default=0.001,
        help="relative spot bump used by the delta comparison",
    )
    for command in (environment, describe, reference, monte_carlo):
        command.add_argument(
            "--output", type=Path, help="save a JSON report, normally under scratch/"
        )
    args = parser.parse_args()

    try:
        if args.command == "environment":
            report = EnvironmentReport.capture().to_json()
        else:
            from .problems import vanilla_case

            case = vanilla_case()
            if args.command == "describe":
                report = case.to_json()
            else:
                from .black_scholes import BlackScholesReference

                reference_value = BlackScholesReference(case.model, case.contract).evaluate(
                    case.initial_spots[0]
                )
                if args.command == "reference":
                    summary = {
                        "method": "black_scholes", "case": case.name,
                        **asdict(reference_value),
                    }
                else:
                    summary = _monte_carlo_summary(args, case, reference_value)
                report = json.dumps(
                    summary, indent=2, sort_keys=True, allow_nan=False,
                )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ArithmeticError) as exc:
        parser.exit(1, f"{args.command.capitalize()} failed: {exc}\n")
    print(report)


if __name__ == "__main__":
    main()
