"""Process isolation, resource limits and JSON records for the locked suite."""

import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _peak_rss_mib() -> float:
    """Use the executed image's high-water mark on Linux and rusage on macOS."""
    if platform.system() == "Linux":
        # Linux rusage can retain a fork/exec startup floor from the controller.
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1])/1024
        raise RuntimeError("Linux process status does not expose VmHWM.")
    if platform.system() == "Darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    raise RuntimeError("Peak resident memory measurement requires macOS or Linux.")


class WorkerLimitError(RuntimeError):
    """An isolated task exceeded its declared wall or resident-memory budget."""


def _child_rss_mib(pid: int) -> float | None:
    if platform.system() == "Linux":
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        except FileNotFoundError:
            return None
        for line in lines:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])/1024
        return None  # A process that has exited can remain briefly as a zombie.
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    if not result.stdout.strip():
        return None
    return float(result.stdout.strip())/1024


def _guarded_worker(command, environment, log, directory, kind, limits):
    started = time.perf_counter()
    peak, status = 0.0, "completed"
    with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT) as process:
        try:
            while process.poll() is None:
                rss = _child_rss_mib(process.pid)
                peak = max(peak, rss if rss is not None else 0)
                if peak > limits["worker_peak_resident_mib"]:
                    status = "memory_limit"
                    break
                if time.perf_counter()-started >= limits["worker_wall_seconds"]:
                    status = "time_limit"
                    break
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        resource_path = directory/f"{kind}_resources.json"
        if resource_path.exists():
            peak = max(peak, json.loads(resource_path.read_text())["peak_resident_mib"])
        if peak > limits["worker_peak_resident_mib"]:
            status = "memory_limit"
        _write_json(directory/f"{kind}_guard.json", {
            "status": status if status != "completed" or process.returncode == 0 else "numerical_failure",
            "peak_resident_mib": peak, "process_wall_seconds": time.perf_counter()-started,
            "returncode": process.returncode, "poll_seconds": 0.2, "limits": limits,
        })
        if status != "completed":
            raise WorkerLimitError(f"{kind}: {status}")
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
