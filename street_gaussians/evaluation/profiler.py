"""Pipeline profiler — hierarchical timing and memory tracking."""

from __future__ import annotations

import time

import torch

from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


def get_gpu_memory_mb() -> tuple[float, float, float] | None:
    """Query current GPU memory usage.

    Returns:
        Tuple of (allocated_MB, reserved_MB, peak_MB), or None if no CUDA.
    """
    if not torch.cuda.is_available():
        return None
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6,
    )


def get_cpu_memory_mb() -> float | None:
    """Query current process RSS memory via psutil or /proc.

    Returns:
        RSS in MB, or None if unavailable.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return None


class PipelineTimer:
    """Hierarchical profiler with CUDA event timing and memory snapshots."""

    def __init__(self, device: str = "cuda") -> None:
        """Initialize the profiler.

        Args:
            device: Device string ("cuda" or "cpu").
        """
        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self._phases: dict[str, list[float]] = {}
        self._active: dict = {}
        self._order: list[str] = []
        self._memory_log: list[dict] = []

    def start(self, name: str) -> None:
        """Begin timing a named phase.

        Args:
            name: Phase identifier.
        """
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._active[name] = event
        else:
            self._active[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """End timing a named phase and record elapsed time.

        Args:
            name: Phase identifier (must match a prior ``start`` call).
        """
        if name not in self._active:
            return
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = self._active[name].elapsed_time(end)
        else:
            elapsed_ms = (time.perf_counter() - self._active[name]) * 1000.0

        if name not in self._phases:
            self._phases[name] = []
            self._order.append(name)
        self._phases[name].append(elapsed_ms)
        del self._active[name]
        self._snapshot_memory(f"after_{name}")

    def _snapshot_memory(self, label: str) -> None:
        """Record a GPU/CPU memory snapshot.

        Args:
            label: Descriptive label for this snapshot.
        """
        gpu = get_gpu_memory_mb()
        cpu = get_cpu_memory_mb()
        self._memory_log.append({
            "label": label,
            "gpu_alloc_mb": gpu[0] if gpu else None,
            "gpu_reserved_mb": gpu[1] if gpu else None,
            "gpu_peak_mb": gpu[2] if gpu else None,
            "cpu_rss_mb": cpu,
        })

    def summary(self) -> str:
        """Generate a formatted profiling summary.

        Returns:
            Multi-line string with timing and memory tables.
        """
        lines = [
            f"\n{'=' * 70}",
            "PIPELINE PROFILING SUMMARY",
            f"{'=' * 70}",
            f"{'Phase':<30s} {'Calls':>6s} {'Total(s)':>10s}"
            f" {'Mean(ms)':>10s} {'Std(ms)':>10s}",
            f"{'-' * 70}",
        ]
        total_all = 0.0
        for name in self._order:
            vals = self._phases[name]
            n = len(vals)
            total_ms = sum(vals)
            mean_ms = total_ms / n if n > 0 else 0
            std_ms = (
                (sum((v - mean_ms) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
                if n > 1
                else 0
            )
            total_s = total_ms / 1000
            total_all += total_s
            lines.append(
                f"{name:<30s} {n:>6d} {total_s:>10.2f}"
                f" {mean_ms:>10.2f} {std_ms:>10.2f}"
            )
        lines.append(f"{'-' * 70}")
        lines.append(f"{'TOTAL':<30s} {'':>6s} {total_all:>10.2f}")

        lines.extend([
            f"\n{'=' * 70}",
            "MEMORY USAGE",
            f"{'=' * 70}",
            f"{'Phase':<35s} {'GPU Alloc(MB)':>14s}"
            f" {'GPU Peak(MB)':>13s} {'CPU RSS(MB)':>12s}",
            f"{'-' * 70}",
        ])
        for snap in self._memory_log:
            gpu_a = (
                f"{snap['gpu_alloc_mb']:.0f}"
                if snap["gpu_alloc_mb"] is not None
                else "N/A"
            )
            gpu_p = (
                f"{snap['gpu_peak_mb']:.0f}"
                if snap["gpu_peak_mb"] is not None
                else "N/A"
            )
            cpu_r = (
                f"{snap['cpu_rss_mb']:.0f}"
                if snap["cpu_rss_mb"] is not None
                else "N/A"
            )
            lines.append(
                f"{snap['label']:<35s} {gpu_a:>14s}"
                f" {gpu_p:>13s} {cpu_r:>12s}"
            )
        lines.append(f"{'=' * 70}")

        return "\n".join(lines)
