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


def _basket_reference_summary(args: argparse.Namespace, case: "PricingCase") -> dict[str, object]:
    from .basket_reference import BasketReferencePricer

    pricer = BasketReferencePricer(case.model, case.contract)
    common = {"pilot_pairs": args.pilot_pairs, "seed": args.seed}
    if args.method == "antithetic":
        result = pricer.antithetic(case.initial_spots, n_pairs=args.pairs, **common)
    else:
        result = pricer.sobol(case.initial_spots, power=args.power, scrambles=args.scrambles, **common)
    return {"case": case.name, "problem": json.loads(case.to_json()), **asdict(result)}


def _bsde_diagnostic_summary(args: argparse.Namespace, case: "PricingCase") -> dict[str, object]:
    import numpy as np

    from .bsde import AnalyticalBSDEDiagnostic, PricingBSDE

    result = AnalyticalBSDEDiagnostic(PricingBSDE(case.model, case.contract)).run(
        case.initial_spots[0], n_paths=args.paths, rng=np.random.default_rng(args.seed),
        steps=tuple(args.steps),
    )
    return {
        "method": "discounted_analytical_bsde_refinement", "case": case.name,
        "problem": json.loads(case.to_json()), "seed": args.seed, **asdict(result),
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
    describe.add_argument("--case", choices=("vanilla", "correlated", "geometric", "geometric-five", "arithmetic-ten"), default="vanilla")
    reference = commands.add_parser(
        "reference", help="evaluate the analytical price and spot Greeks"
    )
    reference.add_argument("--case", choices=("vanilla", "geometric", "geometric-five"), default="vanilla")
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
    basket_reference = commands.add_parser("basket-reference", help="estimate basket price and deltas with geometric controls")
    basket_reference.add_argument("--case", choices=("correlated", "geometric", "vanilla", "arithmetic-ten"), default="correlated")
    basket_reference.add_argument("--method", choices=("antithetic", "sobol"), default="sobol")
    basket_reference.add_argument("--pairs", type=int, default=65_536)
    basket_reference.add_argument("--power", type=int, default=13)
    basket_reference.add_argument("--scrambles", type=int, default=16)
    basket_reference.add_argument("--pilot-pairs", type=int, default=4096)
    basket_reference.add_argument("--seed", type=int, default=2026)
    bsde_diagnostic = commands.add_parser(
        "bsde-diagnostic", help="check the analytical discounted BSDE under time refinement",
    )
    bsde_diagnostic.add_argument("--case", choices=("vanilla",), default="vanilla")
    bsde_diagnostic.add_argument("--paths", type=int, default=8192)
    bsde_diagnostic.add_argument("--seed", type=int, default=2026)
    bsde_diagnostic.add_argument("--steps", type=int, nargs="+", default=(8, 16, 32, 64, 128, 256))
    martingale = commands.add_parser(
        "martingale-control", help="verify zero and analytical martingale controls on IID paths",
    )
    martingale.add_argument("--case", choices=("vanilla", "geometric-five"), default="geometric-five")
    martingale.add_argument("--paths", type=int, default=16_384)
    martingale.add_argument("--seed", type=int, default=2026)
    martingale.add_argument("--steps", type=int, nargs="+", default=(4, 8, 16, 32, 64))
    learned = commands.add_parser(
        "learned-control", help="compare frozen arithmetic-pilot controls with conventional estimates",
    )
    learned.add_argument("--pilot-dir", type=Path, default=Path("scratch/runs/arithmetic_ten_pilot"))
    costs = commands.add_parser(
        "control-cost", help="measure repeated fixed-contract estimates at matched precision",
    )
    costs.add_argument("--pilot-dir", type=Path, default=Path("scratch/runs/arithmetic_ten_pilot"))
    final_suite = commands.add_parser("final-suite", help="execute or resume the locked confirmatory suite")
    final_suite.add_argument("--run-dir", type=Path, default=Path("scratch/runs/final_suite"))
    autograd = commands.add_parser("autograd", help="check physical derivatives and CPU/MPS workloads")
    autograd.add_argument("--backend", choices=("auto", "cpu", "mps"), default="auto")
    autograd.add_argument("--batch-size", type=int, default=256)
    autograd.add_argument("--repeats", type=int, default=5)
    bsde_loss = commands.add_parser("bsde-loss", help="check local BSDE loss against known solutions")
    bsde_loss.add_argument("--case", choices=("vanilla", "geometric"), default="vanilla")
    bsde_loss.add_argument("--paths", type=int, default=2048)
    bsde_loss.add_argument("--seed", type=int, default=2026)
    bsde_loss.add_argument("--steps", type=int, nargs="+", default=(8, 16, 32, 64))
    train = commands.add_parser("train", help="fit the value-gradient solver")
    train.add_argument("--case", choices=("vanilla", "geometric-five", "arithmetic-ten"))
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--iterations", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--terminal-weight", type=float)
    train.add_argument("--patience", type=int, help="stop after this many validation checks without improvement")
    train.add_argument("--stop-after", type=int, help="pause at this absolute update count")
    train.add_argument("--resume", type=Path, help="continue a resume.pt checkpoint with its saved configuration")
    study = commands.add_parser("training-study", help="compare three fixed seeds and terminal weights 1 and 5")
    study.add_argument("--run-dir", type=Path, default=Path("scratch/runs/reproducibility"))
    terminal_train = commands.add_parser("train-terminal", help="fit the terminal-loss Deep BSDE solver")
    terminal_train.add_argument("--case", choices=("vanilla", "geometric-five", "arithmetic-ten"), default="vanilla")
    terminal_train.add_argument("--run-dir", type=Path)
    terminal_train.add_argument("--iterations", type=int, default=3000)
    terminal_train.add_argument("--seed", type=int, default=2026)
    arithmetic = commands.add_parser("arithmetic-pilot", help="run the bounded ten-asset reference and both neural fits")
    arithmetic.add_argument("--run-dir", type=Path, default=Path("scratch/runs/arithmetic_ten_pilot"))
    arithmetic.add_argument("--iterations", type=int, default=3000)
    arithmetic.add_argument("--seed", type=int, default=2026)
    arithmetic.add_argument("--max-seconds", type=float, default=180)
    risk = commands.add_parser("compare-risk", help="compare saved models and recover control-based deltas")
    risk.add_argument("--value-run", type=Path, default=Path("scratch/runs/reproducibility/seed_2026_weight_1"))
    risk.add_argument("--terminal-run", type=Path, default=Path("scratch/runs/vanilla_terminal_bsde"))
    risk.add_argument("--run-dir", type=Path, default=Path("scratch/runs/vanilla_risk_comparison"))
    risk.add_argument("--paths", type=int, default=2048)
    risk.add_argument("--seed", type=int, default=2031)
    design = commands.add_parser("experiment-design", help="validate the specification and export reserved jobs without running them")
    design.add_argument("--spec", type=Path)
    development = commands.add_parser("development-study", help="run the bounded arithmetic development case with three seeds")
    development.add_argument("--case", choices=("arithmetic-25", "arithmetic-50", "arithmetic-100"), default="arithmetic-25")
    development.add_argument("--run-dir", type=Path)
    grid_study = commands.add_parser("grid-study", help="compare the locked time grids at development seeds")
    grid_study.add_argument("--run-dir", type=Path, default=Path("scratch/runs/arithmetic_10_grid_study"))
    stress_study = commands.add_parser("stress-study", help="run the declared financial stresses and Greek diagnostics")
    stress_study.add_argument("--run-dir", type=Path, default=Path("scratch/runs/greek_stress_study"))
    stress_study.add_argument("--geometric-value", type=Path, default=Path("scratch/runs/geometric_five_fbsnn"))
    stress_study.add_argument("--geometric-terminal", type=Path, default=Path("scratch/runs/geometric_five_terminal_bsde"))
    for command in (environment, describe, reference, monte_carlo, finite_difference, sample,
                    basket_reference, bsde_diagnostic, martingale, learned, costs, autograd, bsde_loss, design):
        command.add_argument(
            "--output", type=Path, help="save a JSON report, normally under scratch/"
        )
    args = parser.parse_args()

    try:
        if args.command == "final-suite":
            from .final_suite import FinalSuite

            summary = FinalSuite(args.run_dir).run()
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command in ("learned-control", "control-cost"):
            if args.command == "learned-control":
                from .learned_control import run_learned_control_study

                summary = run_learned_control_study(args.pilot_dir)
            else:
                from .control_cost import ControlCostStudy
                from .experiment_design import ExperimentDesign

                summary = ControlCostStudy.from_pilot(args.pilot_dir, ExperimentDesign.load()).run()
            report = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report+"\n", encoding="utf-8")
            if args.command == "control-cost" and args.output is not None:
                print(json.dumps({"report": str(args.output), "wall_seconds": summary["wall_seconds"],
                                  "matched_precision": summary["matched_precision"]},
                                 indent=2, sort_keys=True, allow_nan=False))
            else:
                print(report)
            return
        if args.command == "stress-study":
            from .greek_study import GreekStressStudy

            result = GreekStressStudy().run(args.run_dir, geometric_value=args.geometric_value,
                                            geometric_terminal=args.geometric_terminal)
            print(json.dumps({"run_directory": str(args.run_dir), "status": result["status"],
                              "specification_sha256": result["specification_sha256"],
                              "scheduled_fits": result["scheduled_fits"], "wall_seconds": result["wall_seconds"],
                              "cases": [{"case": c["case"], "status": c["study"]["status"],
                                         "reference_usable": c["study"]["reference_usable"],
                                         "summary": c["study"]["summary"], "greeks_status": c["greeks"]["status"]}
                                        for c in result["cases"]], "asymmetric_status": result["asymmetric"]["status"]},
                             indent=2, sort_keys=True, allow_nan=False))
            if result["status"] != "completed":
                parser.exit(1, "Greek stress study incomplete; inspect retained task and reference records.\n")
            return
        if args.command in ("development-study", "grid-study"):
            from .development_study import ArithmeticDevelopmentStudy, TimeGridStudy

            if args.run_dir is None:
                args.run_dir = Path("scratch/runs")/f"{args.case.replace('-', '_')}_development"
            study = TimeGridStudy() if args.command == "grid-study" else ArithmeticDevelopmentStudy(args.case)
            result = study.run(args.run_dir)
            print(json.dumps({"run_directory": str(args.run_dir), "status": result["status"],
                              "specification_sha256": result["specification_sha256"],
                              "reference_usable": result["reference_usable"], "reference_levels": result["reference_levels"],
                              "summary": result["summary"], "wall_seconds": result["wall_seconds"]},
                             indent=2, sort_keys=True, allow_nan=False))
            if result["status"] != "completed":
                parser.exit(1, "Development study incomplete; inspect retained task and reference records.\n")
            return
        if args.command == "experiment-design":
            from .experiment_design import ExperimentDesign

            summary = ExperimentDesign.load(args.spec).summary()
            report = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report+"\n")
            print(json.dumps({k: v for k, v in summary.items() if k != "jobs"}, indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command == "arithmetic-pilot":
            from .arithmetic_pilot import ArithmeticPilot, ArithmeticPilotConfig
            from .training import PathTrainingConfig

            result = ArithmeticPilot(ArithmeticPilotConfig(training=PathTrainingConfig(
                iterations=args.iterations, seed=args.seed, max_seconds=args.max_seconds,
            ))).run(args.run_dir)
            print(json.dumps({"run_directory": str(args.run_dir), "status": result["status"],
                              "reference_error_budget": result["reference"]["error_budget_met"],
                              "reference_price": result["reference"]["primary"]["price"],
                              "max_reference_scaled_delta_half_width": max(result["reference"]["reference_scaled_delta_half_widths"]),
                              "models": {k: {
                                  "price": m["initial"]["price"], "price_check": m["price_check"],
                                  "scaled_delta_signed_errors": [d["signed_error"] for d in m["scaled_delta_checks"]],
                                  "scaled_delta_classifications": [d["classification"] for d in m["scaled_delta_checks"]],
                                  "source": m["source"],
                              } for k, m in result["models"].items()},
                              "resources": result["resources"],
                              "budget_projection": result["budget_projection"]},
                             indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command in ("train", "train-terminal"):
            from .problems import arithmetic_ten_case, geometric_five_case, vanilla_case

            case_factory = {"geometric-five": geometric_five_case, "arithmetic-ten": arithmetic_ten_case}.get(args.case, vanilla_case)
            if args.run_dir is None:
                suffix = "fbsnn" if args.command == "train" else "terminal_bsde"
                prefix = {"geometric-five": "geometric_five", "arithmetic-ten": "arithmetic_ten"}.get(args.case, "vanilla")
                args.run_dir = Path("scratch/runs")/f"{prefix}_{suffix}"
        if args.command == "compare-risk":
            from .risk_comparison import compare_saved_runs

            result = compare_saved_runs(args.value_run, args.terminal_run, args.run_dir,
                                        n_paths=args.paths, seed=args.seed)
            print(json.dumps({"run_directory": str(args.run_dir), "models": result.report["models"],
                              "value_gradient_gamma": result.report["value_gradient_gamma"],
                              "targets_met": result.report["targets_met"]},
                             indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command == "train-terminal":
            from .terminal_bsde import train_terminal_and_save
            from .training import PathTrainingConfig

            result = train_terminal_and_save(case_factory(), PathTrainingConfig(
                iterations=args.iterations, seed=args.seed,
            ), args.run_dir)
            print(json.dumps({"run_directory": str(args.run_dir), "status": result["status"],
                              "completed_updates": result["completed_updates"], "selected_step": result["selected_step"],
                              "parameter_count": result["parameter_count"], "training_seconds": result["training_seconds"],
                              "evaluation": result["evaluation"], "targets_met": result["targets_met"]},
                             indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command == "training-study":
            from .problems import vanilla_case
            from .training_study import TrainingStudy

            result = TrainingStudy(vanilla_case()).run(args.run_dir)
            print(json.dumps({"run_directory": str(args.run_dir), "status": result["status"],
                              "resume_check_passed": result["resume_check"]["passed"],
                              "runs": result["runs"], "summary": result["summary"]},
                             indent=2, sort_keys=True, allow_nan=False))
            if result["status"] != "completed":
                parser.exit(1, "Study incomplete; failed or time-limited runs remain in the report.\n")
            return
        if args.command == "train":
            from .training import FBSNNSolver, TrainingConfig, train_and_save

            if args.resume is not None:
                if any(value is not None for value in (args.iterations, args.seed, args.terminal_weight, args.patience)):
                    raise ValueError("Resume uses the saved configuration; omit training-configuration flags.")
                if args.run_dir.exists():
                    raise FileExistsError(f"Run directory already exists: {args.run_dir}")
                solver = FBSNNSolver.resume(args.resume)
                if args.case is not None and args.case != solver.case.name:
                    raise ValueError("Case flag disagrees with the resume checkpoint.")
                result = solver.fit(stop_after=args.stop_after)
                solver.save(args.run_dir, result)
            else:
                config = {name: value for name, value in (
                    ("iterations", args.iterations), ("seed", args.seed),
                    ("terminal_weight", args.terminal_weight), ("patience", args.patience),
                ) if value is not None}
                result = train_and_save(case_factory(), TrainingConfig(**config),
                                        args.run_dir, stop_after=args.stop_after)
            print(json.dumps({
                "run_directory": str(args.run_dir), "status": result["status"],
                "completed_updates": result["completed_updates"],
                "training_seconds": result["training_seconds"],
                "evaluation": result["evaluation"], "targets_met": result["targets_met"],
            }, indent=2, sort_keys=True, allow_nan=False))
            return
        if args.command == "environment":
            report = EnvironmentReport.capture().to_json()
        elif args.command == "autograd":
            from .autograd_diagnostics import run_autograd_diagnostics

            summary = run_autograd_diagnostics(
                backend=args.backend, batch_size=args.batch_size, repeats=args.repeats,
            )
            report = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
        else:
            from .problems import arithmetic_ten_case, correlated_case, geometric_case, geometric_five_case, vanilla_case

            case = {"vanilla": vanilla_case, "correlated": correlated_case,
                    "geometric": geometric_case, "geometric-five": geometric_five_case,
                    "arithmetic-ten": arithmetic_ten_case}[args.case]()
            if args.command == "describe":
                report = case.to_json()
            else:
                if args.command == "sample":
                    summary = _sampling_summary(args, case)
                elif args.command == "basket-reference":
                    summary = _basket_reference_summary(args, case)
                elif args.command == "bsde-diagnostic":
                    summary = _bsde_diagnostic_summary(args, case)
                elif args.command == "martingale-control":
                    from .martingale import run_known_control_study

                    summary = run_known_control_study(
                        case, n_paths=args.paths, seed=args.seed, steps=tuple(args.steps),
                    )
                    summary["problem"] = json.loads(case.to_json())
                elif args.command == "bsde-loss":
                    from .loss_diagnostics import run_loss_diagnostics

                    summary = run_loss_diagnostics(
                        case, n_paths=args.paths, seed=args.seed, steps=tuple(args.steps),
                    )
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
    if args.command in ("autograd", "bsde-loss") and not summary["passed"]:
        parser.exit(1, f"{args.command.capitalize()} checks failed; inspect the report above.\n")


if __name__ == "__main__":
    main()
