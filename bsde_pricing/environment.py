"""Capture the numerical runtime and verify its CPU dependencies."""

from dataclasses import asdict, dataclass
import importlib
import json
import os
import platform
import subprocess
import sys


def _physical_memory_gib() -> float | None:
    """Return installed physical memory where the operating system exposes it."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            memory_bytes = int(result.stdout.strip())
        else:
            memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError):
        return None
    return round(memory_bytes / 1024**3, 2)


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    executable: str
    operating_system: str
    architecture: str
    physical_memory_gib: float | None
    package_versions: dict[str, str]
    cpu_float64_checks: dict[str, float]
    mps_built: bool
    mps_available: bool

    @classmethod
    def capture(cls) -> "EnvironmentReport":
        modules = {}
        for name in ("numpy", "scipy", "matplotlib", "pytest", "torch"):
            try:
                modules[name] = importlib.import_module(name)
            except ImportError as exc:
                raise RuntimeError(
                    f"Cannot import {name}: {exc}. "
                    "Activate the environment defined in environment.yml."
                ) from exc

        numpy = modules["numpy"]
        torch = modules["torch"]
        size = 256
        numpy_matrix = numpy.ones((size, size), dtype=numpy.float64)
        torch_matrix = torch.ones(
            (size, size), dtype=torch.float64, device="cpu"
        )
        cpu_checks = {
            "numpy_matmul_sum": float((numpy_matrix @ numpy_matrix).sum()),
            "torch_matmul_sum": (torch_matrix @ torch_matrix).sum().item(),
        }
        expected = float(size**3)
        for name, value in cpu_checks.items():
            if value != expected:
                raise RuntimeError(
                    f"CPU float64 check {name} returned {value}; expected {expected}."
                )

        os_version = (
            platform.mac_ver()[0]
            if platform.system() == "Darwin"
            else platform.release()
        )
        return cls(
            python=platform.python_version(),
            executable=sys.executable,
            operating_system=f"{platform.system()} {os_version}",
            architecture=platform.machine(),
            physical_memory_gib=_physical_memory_gib(),
            package_versions={
                name: module.__version__ for name, module in modules.items()
            },
            cpu_float64_checks=cpu_checks,
            mps_built=torch.backends.mps.is_built(),
            mps_available=torch.backends.mps.is_available(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
