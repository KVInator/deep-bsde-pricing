"""Validate the declared experiment and enumerate jobs without sampling or fitting."""

import hashlib
import json
from pathlib import Path

from .problems import EuropeanCall, GBMModel, PricingCase, geometric_five_case, vanilla_case


class ExperimentDesign:
    """One specification owns case inputs, reserved seeds and planned workloads."""

    def __init__(self, data: dict):
        self._data = json.loads(json.dumps(data, allow_nan=False))
        try:
            self._validate()
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete or incorrectly typed experiment specification.") from exc

    @classmethod
    def load(cls, path: Path | None = None) -> "ExperimentDesign":
        path = Path(__file__).resolve().parent.parent/"experiments/design.json" if path is None else Path(path)
        return cls(json.loads(path.read_text()))

    @property
    def data(self) -> dict:
        return json.loads(json.dumps(self._data))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self._data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def case(self, name: str) -> PricingCase:
        row = next((c for c in self._data["cases"] if c["name"] == name), None)
        if row is None:
            raise ValueError(f"Unknown experiment case: {name}")
        if row["family"] == "vanilla":
            return vanilla_case()
        if row["family"] == "geometric":
            return geometric_five_case()
        base = {**self._data["base_arithmetic"], **row}
        d = row["dimension"]
        return PricingCase(name, GBMModel(base["rate"], (base["volatility"],)*d, (base["dividend"],)*d,
                           tuple(tuple(1.0 if i == j else base["correlation"] for j in range(d)) for i in range(d))),
                           EuropeanCall(base["strike"], base["maturity"], (1/d,)*d), (base["spot"],)*d)

    def _validate(self) -> None:
        data = self._data
        required = {"version", "status", "cases", "base_arithmetic", "training", "grid_case", "grid_steps",
                    "solvers", "development_seeds", "final_training_seeds", "final_streams", "targets",
                    "reference", "resources", "tuning", "estimators"}
        if set(data) != required or data["version"] != 1:
            raise ValueError("Unsupported or incomplete experiment specification.")
        if data["status"] != "locked_before_confirmatory_runs":
            raise ValueError("The specification must declare its lock status.")
        names = [c["name"] for c in data["cases"]]
        if len(names) != len(set(names)) or not names or data["grid_case"] not in names:
            raise ValueError("Case names must be unique and include the grid case.")
        for row in data["cases"]:
            if row["family"] not in ("vanilla", "geometric", "arithmetic"):
                raise ValueError("Unsupported case family.")
            identity = {"name", "family", "dimension"}
            if row["family"] == "arithmetic":
                if not set(row) <= identity | {"strike", "maturity", "correlation"}:
                    raise ValueError("Unknown arithmetic-case override.")
            elif (set(row) != identity or row["name"] !=
                  {"vanilla": "vanilla", "geometric": "geometric-five"}[row["family"]]):
                raise ValueError("Analytical cases use their existing canonical financial specification.")
            d = row["dimension"]
            if isinstance(d, bool) or not isinstance(d, int) or d < 1:
                raise ValueError("Case dimension must be a positive integer.")
            case = self.case(row["name"])
            if case.model.dimension != d:
                raise ValueError("Case dimension disagrees with its financial specification.")
        if data["solvers"] != ["value_gradient", "terminal_loss"]:
            raise ValueError("Both declared solvers are required.")
        development, final = data["development_seeds"], data["final_training_seeds"]
        streams = [base+i for base in data["final_streams"].values() for i in range(len(names))]
        all_seeds = development+final+streams
        if (any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in all_seeds)
                or len(set(all_seeds)) != len(all_seeds) or len(final) != 3):
            raise ValueError("Development, final training and reserved evaluation seeds must be disjoint.")
        steps = data["grid_steps"]
        if (not steps or len(set(steps)) != len(steps) or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in steps)
                or data["training"]["base_steps"] in steps):
            raise ValueError("Additional grids must be unique positive intervals distinct from the base grid.")
        from .training import TrainingConfig

        training = dict(data["training"])
        training["n_steps"] = training.pop("base_steps")
        TrainingConfig(**training, seed=final[0])
        targets = data["targets"]
        if (targets["price_absolute"] <= 0 or targets["weight_scaled_delta_absolute"] <= 0 or
                not 0 < targets["reference_fraction"] <= 1 or targets["confidence_level"] != 0.95):
            raise ValueError("Invalid accuracy targets or reference allocation.")
        if data["tuning"]["final_accuracy_selects_checkpoint"] or any(
                data["tuning"][k] != 0 for k in ("additional_architectures", "additional_hyperparameter_trials")):
            raise ValueError("The locked design permits no further tuning or final-error selection.")
        ref = data["reference"]
        if (not ref["sobol_powers"] or any(isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 20
                for p in ref["sobol_powers"]) or any(isinstance(n, bool) or not isinstance(n, int) or n < 2
                for n in ref["check_pairs"])):
            raise ValueError("Invalid reference sample allocations.")
        if (ref["sobol_powers"] != sorted(set(ref["sobol_powers"])) or
                len(ref["sobol_powers"]) != len(ref["check_pairs"]) or ref["scrambles"] < 4):
            raise ValueError("Reference refinement levels and independent checks must align.")
        for key in ("worker_wall_seconds", "worker_peak_resident_mib", "evaluation_chunk_paths", "final_total_wall_hours"):
            if data["resources"][key] <= 0:
                raise ValueError("Resource limits must be positive.")
        if data["resources"]["worker_wall_seconds"] <= training["max_seconds"] or not data["resources"]["sequential_workers"]:
            raise ValueError("Sequential workers need a wall limit above the fitting cap.")

    def jobs(self) -> list[dict]:
        jobs = []
        for row in self._data["cases"]:
            grids = [self._data["training"]["base_steps"]]
            if row["name"] == self._data["grid_case"]:
                grids += self._data["grid_steps"]
            for steps in grids:
                for solver in self._data["solvers"]:
                    for seed in self._data["final_training_seeds"]:
                        jobs.append({"case": row["name"], "n_steps": steps, "solver": solver, "seed": seed})
        return jobs

    def summary(self) -> dict:
        jobs = self.jobs()
        return {"status": self._data["status"], "specification_sha256": self.sha256,
                "financial_cases": len(self._data["cases"]), "planned_neural_fits": len(jobs),
                "fit_soft_cap_hours": len(jobs)*self._data["training"]["max_seconds"]/3600,
                "worker_wall_cap_hours": len(jobs)*self._data["resources"]["worker_wall_seconds"]/3600,
                "targets": self._data["targets"], "resources": self._data["resources"],
                "final_samples_drawn": 0, "training_started": False,
                "execution_support": "design validation and job enumeration; execution belongs to the later suite",
                "jobs": jobs}
