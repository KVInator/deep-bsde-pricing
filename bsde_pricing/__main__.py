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
    for command in (environment, describe, reference):
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

                result = BlackScholesReference(case.model, case.contract).evaluate(
                    case.initial_spots[0]
                )
                report = json.dumps(
                    {"method": "black_scholes", "case": case.name, **asdict(result)},
                    indent=2, sort_keys=True,
                )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ArithmeticError) as exc:
        parser.exit(1, f"{args.command.capitalize()} failed: {exc}\n")
    print(report)


if __name__ == "__main__":
    main()
