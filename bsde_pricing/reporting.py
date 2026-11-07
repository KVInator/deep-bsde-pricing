"""Extract the notebook tables from a final-suite ledger."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from .control_cost import _matched_precision, _summarise


def _reference(row):
    successful = [level["result"] for level in row["levels"] if level["status"] == "completed"]
    summary = {"status": row["status"], "levels": [level["status"] for level in row["levels"]]}
    if "primary" in row:
        summary.update(row["primary"])
    if successful:
        result = successful[-1]
        summary.update(
            sobol_power=result["sobol_levels"][-1]["power"],
            price_half_width=result["reference_price_half_width"],
            max_scaled_delta_half_width=max(result["reference_scaled_delta_half_widths"]),
            max_combined_se_gap=max(abs(check["difference_in_standard_errors"] or 0)
                                    for check in result["independent_method_agreement"]),
        )
    return summary


def _fit(row):
    result = row.get("result", {})
    price_check = result.get("price_check") or {}
    delta_checks = result.get("delta_checks", [])
    values = {key: value for key, value in result.items() if key not in {"price_check", "delta_checks"}}
    return {
        "job": row["job"], "status": row["status"], **values,
        "worker_peak_mib": row.get("attempts", [{}])[-1].get("guard", {}).get("peak_resident_mib"),
        "price_error": price_check.get("signed_error"),
        "scaled_delta_errors": [check["signed_error"] for check in delta_checks],
        "price_classification": price_check.get("classification"),
        "delta_classifications": dict(Counter(check["classification"] for check in delta_checks)),
    }


def _estimator_cohort(rows, fit_results, reference, specification):
    counts = dict(Counter(row["status"] for row in rows))
    summary = {"status_counts": counts}
    if counts != {"completed": len(rows)}:
        return summary

    allocations = []
    protocol = specification["estimators"]
    for pairs in protocol["antithetic_pairs"]:
        estimates = [row["result"]["methods_by_pairs"][str(pairs)] for row in rows]
        allocations.append({"pairs": pairs, "methods": {
            method: _summarise([record[method] for record in estimates], reference)
            for method in estimates[0]
        }})

    fixed = {}
    for method in allocations[0]["methods"]:
        solver = method.removesuffix("_unit").removesuffix("_fitted")
        setup = [row["result"]["checkpoint_setup_seconds"].get(solver, 0.0) for row in rows]
        fixed[method] = {
            "training_seconds": fit_results.get(solver, {}).get("training_seconds", 0.0),
            "checkpoint_setup_seconds": median(setup),
        }
    summary.update(allocations=allocations, fixed_costs=fixed,
                   matched_precision=_matched_precision(allocations, fixed, len(rows)))
    return summary


def summarize(report: dict) -> dict:
    manifest = report["manifest"]
    specification = manifest["specification"]
    references = {name: _reference(row) for name, row in report["references"].items()}
    fits = [_fit(row) for row in report["fits"]]
    estimator_case = specification["estimators"]["reference_case"]
    estimator_grid = specification["training"]["base_steps"]
    estimators = {}
    for seed in specification["final_training_seeds"]:
        rows = [row for row in report["estimators"] if row["seed"] == seed]
        seed_fits = {row["job"]["solver"]: row for row in fits
                     if (row["job"]["case"], row["job"]["n_steps"], row["job"]["seed"])
                     == (estimator_case, estimator_grid, seed)}
        reference = references[estimator_case].get("price", {}).get("estimate")
        estimators[str(seed)] = _estimator_cohort(rows, seed_fits, reference, specification)

    return {
        "runtime": manifest["runtime"], "specification_sha256": manifest["specification_sha256"],
        "source_sha256": manifest["source_sha256"], "elapsed_seconds": report["elapsed_seconds"],
        "complete": "summary" in report, "references": references, "fits": fits, "estimators": estimators,
    }


def write_summary(report: Path, output: Path) -> None:
    if report.resolve() == output.resolve():
        raise ValueError("The output must differ from the source ledger.")
    source = report.read_bytes()
    summary = summarize(json.loads(source))
    summary["report_sha256"] = hashlib.sha256(source).hexdigest()
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded+"\n", encoding="utf-8")
