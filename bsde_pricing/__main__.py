"""Command-line entry point for pricing experiments and runtime diagnostics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .environment import EnvironmentReport


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
                    import numpy as np

                    from .monte_carlo import MonteCarloPricer

                    rng = np.random.default_rng(args.seed)
                    result = MonteCarloPricer(case.model, case.contract).estimate(
                        case.initial_spots, n_paths=args.paths, rng=rng
                    )
                    error = result.price_estimate - reference_value.call_price
                    summary = {
                        "method": "exact_terminal_monte_carlo", "case": case.name,
                        "seed": args.seed,
                        "bit_generator": type(rng.bit_generator).__name__,
                        **asdict(result),
                        "reference_call_price": reference_value.call_price,
                        "signed_error": error,
                        "error_in_standard_errors": (
                            error / result.standard_error
                            if result.standard_error > 0 else None
                        ),
                    }
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
