"""Command-line entry point for pricing experiments and runtime diagnostics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .environment import EnvironmentReport

if TYPE_CHECKING:
    from .black_scholes import BlackScholesResult
    from .geometric_basket import GeometricBasketResult
    from .problems import PricingCase


def _monte_carlo_summary(
    args: argparse.Namespace, case: "PricingCase",
    reference_value: "BlackScholesResult | GeometricBasketResult",
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
        if case.contract.basket == "geometric":
            delta_result = pricer.estimate_coordinate_delta(
                case.initial_spots, coordinate=args.coordinate, n_paths=args.paths,
                rng=rng, bump_fraction=args.bump_fraction,
            )
            reference_delta = reference_value.call_deltas[args.coordinate]
            summary["coordinate"] = args.coordinate
        else:
            if args.coordinate != 0:
                raise ValueError("The vanilla case has only coordinate zero.")
            delta_result = pricer.estimate_delta(
                case.initial_spots, n_paths=args.paths, rng=rng,
                bump_fraction=args.bump_fraction,
            )
            reference_delta = reference_value.call_delta
        if reference_delta is None:
            raise ValueError("The case has no defined analytical spot delta.")
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


def _finite_difference_summary(
    args: argparse.Namespace, case: "PricingCase", reference_value: "BlackScholesResult",
) -> dict[str, object]:
    from .finite_difference import FiniteDifferencePricer

    result = FiniteDifferencePricer(case.model, case.contract).solve(
        s_max=args.s_max, n_space=args.space_steps, n_time=args.time_steps,
        rannacher_steps=args.rannacher_steps,
    )
    value = result.evaluate(case.initial_spots[0])
    return {
        "case": case.name,
        "method": "crank_nicolson_rannacher" if args.rannacher_steps else "crank_nicolson",
        **asdict(value),
        "s_max": args.s_max, "space_steps": args.space_steps,
        "time_steps": args.time_steps, "rannacher_steps": args.rannacher_steps,
        "spot_spacing": float(result.spots[1]),
        "time_spacing": result.time_to_maturity / result.n_time,
        "upwind_node_count": result.upwind_node_count,
        "reference_call_price": reference_value.call_price,
        "reference_call_delta": reference_value.call_delta,
        "reference_gamma": reference_value.gamma,
        "price_signed_error": value.call_price - reference_value.call_price,
        "delta_signed_error": value.call_delta - reference_value.call_delta,
        "gamma_signed_error": value.gamma - reference_value.gamma,
    }


def _sampling_summary(args: argparse.Namespace, case: "PricingCase") -> dict[str, object]:
    import numpy as np

    from .sampling import TerminalSampleDiagnostics

    if args.paths < 2:
        raise ValueError("Sampling diagnostics require at least two paths.")
    horizon = case.contract.maturity if args.horizon is None else args.horizon
    rng = np.random.default_rng(args.seed)
    terminal = case.model.sample_terminal(case.initial_spots, horizon, args.paths, rng)
    diagnostics = TerminalSampleDiagnostics.from_sample(
        case.model, case.initial_spots, horizon, terminal,
    )
    return {
        "method": "exact_correlated_terminal_gbm", "case": case.name,
        "seed": args.seed, "bit_generator": type(rng.bit_generator).__name__,
        "dimension": case.model.dimension, "initial_spots": case.initial_spots,
        "rate": case.model.rate, "dividend_yields": case.model.dividend_yields,
        "volatilities": case.model.volatilities, "correlation": case.model.correlation,
        **asdict(diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bsde_pricing")
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser(
        "environment", help="check numerical dependencies and report the runtime"
    )
    describe = commands.add_parser(
        "describe", help="show a pricing case and its financial conventions"
    )
    describe.add_argument("--case", choices=("vanilla", "correlated", "geometric"), default="vanilla")
    reference = commands.add_parser(
        "reference", help="evaluate the analytical price and spot Greeks"
    )
    reference.add_argument("--case", choices=("vanilla", "geometric"), default="vanilla")
    monte_carlo = commands.add_parser(
        "monte-carlo", help="estimate a price and its uncertainty from IID terminal states"
    )
    monte_carlo.add_argument("--case", choices=("vanilla", "geometric"), default="vanilla")
    monte_carlo.add_argument("--paths", type=int, default=100_000)
    monte_carlo.add_argument("--seed", type=int, default=2026)
    monte_carlo.add_argument("--quantity", choices=("price", "delta"), default="price")
    monte_carlo.add_argument("--coordinate", type=int, default=0,
                             help="zero-based asset index for a delta calculation")
    monte_carlo.add_argument(
        "--bump-fraction", type=float, default=0.001,
        help="relative spot bump used by the delta comparison",
    )
    finite_difference = commands.add_parser(
        "finite-difference", help="solve the scalar call PDE and compare price and Greeks"
    )
    finite_difference.add_argument("--case", choices=("vanilla",), default="vanilla")
    finite_difference.add_argument("--s-max", type=float, default=400.0)
    finite_difference.add_argument("--space-steps", type=int, default=400)
    finite_difference.add_argument("--time-steps", type=int, default=400)
    finite_difference.add_argument("--rannacher-steps", type=int, default=2)
    sample = commands.add_parser("sample", help="check scalar or correlated terminal GBM moments")
    sample.add_argument("--case", choices=("vanilla", "correlated"), default="correlated")
    sample.add_argument("--paths", type=int, default=100_000)
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--horizon", type=float, help="simulation horizon in years; default is maturity")
    for command in (environment, describe, reference, monte_carlo, finite_difference, sample):
        command.add_argument(
            "--output", type=Path, help="save a JSON report, normally under scratch/"
        )
    args = parser.parse_args()

    try:
        if args.command == "environment":
            report = EnvironmentReport.capture().to_json()
        else:
            from .problems import correlated_case, geometric_case, vanilla_case

            case = {"vanilla": vanilla_case, "correlated": correlated_case,
                    "geometric": geometric_case}[args.case]()
            if args.command == "describe":
                report = case.to_json()
            else:
                if args.command == "sample":
                    summary = _sampling_summary(args, case)
                else:
                    from .black_scholes import BlackScholesReference
                    from .geometric_basket import GeometricBasketReference

                    reference_value = (
                        GeometricBasketReference(case.model, case.contract).evaluate(case.initial_spots)
                        if case.contract.basket == "geometric" else
                        BlackScholesReference(case.model, case.contract).evaluate(case.initial_spots[0])
                    )
                    if args.command == "reference":
                        summary = {
                            "method": ("geometric_basket" if case.contract.basket == "geometric"
                                       else "black_scholes"), "case": case.name,
                            **asdict(reference_value),
                        }
                    elif args.command == "monte-carlo":
                        summary = _monte_carlo_summary(args, case, reference_value)
                    else:
                        summary = _finite_difference_summary(args, case, reference_value)
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
