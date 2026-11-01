"""Optional, CPU-only law-moment sensitivity illustration; no timegram simulator/training.

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/identification/sensitivity.py
Numerical records go to experiments/identification/results; --figures selects a destination
for the two figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "timegram.moment-sensitivity/1"
T = 128
RTOLS = (1e-12, 1e-9, 1e-6, 1e-3)
ATOL = 1e-12
STEPS = (1e-3, 1e-4, 1e-5, 1e-6)
DELTAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3)
CASES = {"separated": (0.2, 0.8, 1.0, 1.0), "zero_phi": (0.0, 0.6, 1.0, 1.0)}
CASES.update({f"delta_{delta:g}": (0.5, 0.5 + delta, 1.0, 1.0) for delta in DELTAS})


def setup_threads():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def _coordinates(theta, length):
    if theta.shape != (4,) or theta.dtype != torch.float64 or theta.device.type != "cpu":
        raise ValueError("theta must be a CPU float64 vector (phi1, phi2, v1, v2)")
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("history length must be a positive integer")


def validate_point(theta, length=T):
    _coordinates(theta, length)
    if not torch.isfinite(theta).all() or not (theta[:2].abs() < 1).all() or not (theta[2:] > 0).all():
        raise ValueError("point must have finite |phi_i|<1 and positive innovation variances")


def lower_filter(phi, length):
    # A recurrence avoids negative powers or 0**(-1) in the derivative at phi=0.
    powers = [phi.new_ones(())]
    for _ in range(1, length):
        powers.append(powers[-1] * phi)
    lag = torch.arange(length)[:, None] - torch.arange(length)[None, :]
    return torch.stack(powers)[lag.clamp_min(0)] * (lag >= 0)


def covariance(theta, length=T):
    """Exact real-law formula evaluated in float64; x_i,0=0, t=1,...,length."""
    _coordinates(theta, length)
    a, b = lower_filter(theta[0], length), lower_filter(theta[1], length)
    return theta[2] * (a @ a.T) + theta[3] * (b @ b.T)


def moment_map(theta, length=T):
    cov = covariance(theta, length)
    rows, cols = torch.tril_indices(length, length)
    return torch.cat((theta.new_zeros(length), cov[rows, cols]))


def analytic_jacobian(theta, length=T):
    _coordinates(theta, length)
    rows, cols = torch.tril_indices(length, length)
    lag = torch.arange(length)[:, None] - torch.arange(length)[None, :]
    kernels, derivatives = [], []
    for i in range(2):
        phi = theta[i]
        l = lower_filter(phi, length)
        derivative_powers = torch.stack([phi.new_zeros(())] +
                                       [j * phi ** (j - 1) for j in range(1, length)])
        dl = derivative_powers[lag.clamp_min(0)] * (lag >= 0)
        kernels.append(l @ l.T)
        derivatives.append(theta[i + 2] * (dl @ l.T + l @ dl.T))
    return torch.stack([torch.cat((theta.new_zeros(length), c[rows, cols]))
                        for c in derivatives + kernels], dim=1)


def fixed_row_scale(theta0, length=T):
    """A fixed Euclidean moment scaling, not a Fisher/likelihood whitening metric."""
    variance0 = covariance(theta0, length)[0, 0].detach()
    return torch.cat((variance0.rsqrt().expand(length),
                      variance0.reciprocal().expand(length * (length + 1) // 2)))


def numerical_rank(singular_values, rtol=1e-9, atol=ATOL):
    largest = float(singular_values.max()) if singular_values.numel() else 0.0
    return int((singular_values > max(atol, rtol * largest)).sum())


def error_summary(actual, expected):
    norm = float(torch.linalg.vector_norm(expected))
    error = actual - expected
    absolute = float(error.abs().max())
    return {"max_absolute": absolute,
            "relative_frobenius": float(torch.linalg.vector_norm(error)) / norm if norm else (0.0 if absolute == 0 else None)}


def central_difference(theta0, step, scale=None, length=T):
    scale = torch.eye(4, dtype=torch.float64) if scale is None else scale
    columns = []
    for direction in scale.T:
        plus, minus = theta0 + step * direction, theta0 - step * direction
        validate_point(plus, length)
        validate_point(minus, length)
        columns.append((moment_map(plus, length) - moment_map(minus, length)) / (2 * step))
    return torch.stack(columns, dim=1)


try:                                  # Unix only
    import resource
except ImportError:
    resource = None


def _peak_rss():
    """Own-process high-water resident memory: (native value, native unit, bytes).
    Unix: getrusage. Windows: psutil's peak working set when psutil is installed.
    Otherwise (None, "unavailable", None); it never pretends to measure instantaneous RSS."""
    if resource is not None:
        native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        unit = "bytes" if sys.platform == "darwin" else "KiB"
        return native, unit, int(native * (1 if sys.platform == "darwin" else 1024))
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except ImportError:
        peak = None
    return (peak, "bytes (peak working set)", peak) if peak is not None else (None, "unavailable", None)


def _rss_bytes():
    return _peak_rss()[2]


def _spectrum(jacobian):
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    return {"singular_values": singular.tolist(),
            "rank_by_rtol": {f"{r:g}": numerical_rank(singular, r) for r in RTOLS},
            "right_singular_vectors_scaled_coordinates": vh.T.tolist(),
            "right_singular_vectors_original_coordinates": vh.T.tolist()}, vh


def run_case(name, output):
    """One fresh subprocess per measured point; no cached derivative setup is reused."""
    setup_threads()
    theta0 = torch.tensor(CASES[name], dtype=torch.float64)
    validate_point(theta0)
    scale = torch.eye(4, dtype=torch.float64)
    baseline_rss = _rss_bytes()
    started = time.perf_counter()
    moments = moment_map(theta0)
    moment_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jacobian = torch.func.jacfwd(lambda u: moment_map(theta0 + scale @ u))(torch.zeros_like(theta0))
    jacobian_seconds = time.perf_counter() - started
    row_scale = fixed_row_scale(theta0)
    scaled = row_scale[:, None] * jacobian
    started = time.perf_counter()
    raw_spectrum, raw_vh = _spectrum(jacobian)
    scaled_spectrum, scaled_vh = _spectrum(scaled)
    svd_seconds = time.perf_counter() - started
    analytic = analytic_jacobian(theta0)
    analytic_error = error_summary(jacobian, analytic)
    if not torch.allclose(jacobian, analytic, atol=1e-10, rtol=1e-9):
        raise AssertionError(f"autodiff/analytic disagreement: {analytic_error}")
    fd = {f"{h:g}": error_summary(central_difference(theta0, h, scale), jacobian) for h in STEPS}
    if min(x["relative_frobenius"] for x in fd.values()) > 1e-6:
        raise AssertionError(f"finite differences did not agree: {fd}")
    peak_native, native_unit, peak_bytes = _peak_rss()
    variance0 = float(moments[T])
    record = {"schema": SCHEMA, "case": name, "theta": theta0.tolist(), "length": T,
              "parameter_order": ["phi1", "phi2", "innovation_variance1", "innovation_variance2"],
              "coordinate_units": ["dimensionless", "dimensionless", "observation_units_squared", "observation_units_squared"],
              "domain": {"phi1": "(-1,1)", "phi2": "(-1,1)", "v1": "(0,infinity)", "v2": "(0,infinity)",
                         "scope": "fixed finite raw history with zero initial states; no random-parameter mixture"},
              "coordinate_scale": scale.tolist(), "moment_scaling": {"mean_divisor": variance0**0.5,
                    "covariance_divisor": variance0, "held_fixed_during_differentiation": True},
              "initialization": "both latent components exactly zero at t=0",
              "observation": "raw complete history t=1..128; mean and row-major lower covariance triangle",
              "metric": "Euclidean moments, not Fisher information", "dtype": "float64", "device": "cpu",
              "method": "torch.func.jacfwd", "rank_atol": ATOL, "rank_rtols": RTOLS,
              "raw": raw_spectrum, "fixed_scaled": scaled_spectrum,
              "analytic_error": analytic_error, "finite_difference_errors": fd,
              "timings_seconds": {"moments": moment_seconds, "jacfwd_including_first_call_setup": jacobian_seconds,
                                  "two_svd_decompositions": svd_seconds},
              "memory": {"baseline_process_peak_rss_bytes": baseline_rss, "peak_process_rss_bytes": peak_bytes,
                         "instantaneous_baseline_rss_bytes": None,
                         "baseline_scope": "pre-measurement OS high water including imports; instantaneous RSS not collected",
                         "peak_rss_native": peak_native, "native_units": native_unit,
                         "tensor_storage_bytes_retained_lower_bound": sum(x.numel()*x.element_size() for x in
                              (moments, jacobian, analytic, scaled, row_scale, raw_vh, scaled_vh)),
                         "scope": "whole fresh subprocess high water including imports/checks; not Torch allocator peak"},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "intraop_threads": torch.get_num_threads(),
                              "interop_threads": torch.get_num_interop_threads()}}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(output / f"{name}.npz", theta=theta0.numpy(), moments=moments.numpy(),
                        jacobian=jacobian.numpy(), analytic=analytic.numpy(), fixed_scaled_jacobian=scaled.numpy(),
                        row_scale=row_scale.numpy(), raw_right_vectors=raw_vh.T.numpy(),
                        fixed_scaled_right_vectors=scaled_vh.T.numpy())
    return record


def counterexamples(output=None):
    """Separately label analytic witnesses, numerical sensitivity and path derivatives."""
    theta = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float64)
    redistributed = torch.tensor([0.5, 0.5, 1.5, 0.5], dtype=torch.float64)
    delta = 0.01
    opposite = theta + torch.tensor([delta, -delta, 0, 0], dtype=torch.float64)
    cov, redistributed_cov, opposite_cov = [covariance(x) for x in (theta, redistributed, opposite)]
    separated = torch.tensor(CASES["separated"], dtype=torch.float64)
    swap_error = float((covariance(separated) - covariance(separated[[1, 0, 3, 2]])).abs().max())
    n, seed = 128, 20261029
    noise = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    std = torch.tensor([1.0, 2.0], dtype=torch.float64)
    triangular = torch.eye(n, dtype=torch.float64)[torch.tril_indices(n,n).unbind()]
    law_variance = torch.func.jacfwd(lambda v: v.sum() * triangular)(std.square())
    law_scale = torch.func.jacfwd(lambda s: s.square().sum() * triangular)(std)
    path_scale = torch.func.jacfwd(lambda s: noise @ s)(std)
    spectra = {name: torch.linalg.svdvals(j).tolist() for name, j in
               (("law_variance_coordinates", law_variance), ("law_scale_coordinates", law_scale),
                ("fixed_path_scale_coordinates", path_scale))}
    cube = lambda x: torch.stack((x**3, x.new_ones(())))
    cube_zero = torch.func.jacfwd(cube)(torch.tensor(0.0, dtype=torch.float64))
    cube_one = torch.func.jacfwd(cube)(torch.tensor(1.0, dtype=torch.float64))
    record = {"variance_witness": {"theta": theta.tolist(), "alternative": redistributed.tolist(),
                   "analytic_argument": "same phi, zero starts, independent Gaussian shocks: C=(v1+v2)K(phi)",
                   "scope": "exact real conditional law for every positive finite length; numerical replay at T=128",
                   "numerical_max_covariance_error": float((cov-redistributed_cov).abs().max())},
              "persistence_null_is_not_a_witness": {"delta": delta, "alternative": opposite.tolist(),
                   "max_covariance_change": float((cov-opposite_cov).abs().max()),
                   "lag_two_entry_change": float(opposite_cov[2,0]-cov[2,0]),
                   "analytic_lag_two_change": 2*delta**2},
              "label_swap": {"numerical_max_covariance_error": swap_error,
                   "scope": "permutation of the two independent component labels; full rank is not global identification"},
              "path_vs_law": {"seed": seed, "length": n, "scales": std.tolist(),
                   "variances": std.square().tolist(), "spectra": spectra,
                   "ranks_at_rtol_1e-9": {k: numerical_rank(torch.tensor(v, dtype=torch.float64), 1e-9) for k,v in spectra.items()},
                   "scope": "fixed-noise path rank is not observation-law rank; positive scale/variance coordinate maps are invertible"},
              "injective_cube": {"derivative_at_zero": cube_zero.tolist(), "derivative_at_one": cube_one.tolist(),
                   "argument": "theta -> theta^3 is strictly increasing over R; zero derivative is first-order insensitivity, not a nonidentification witness"}}
    if output is not None:
        (output / "counterexamples.json").write_text(json.dumps(record, indent=2) + "\n")
        np.savez_compressed(output / "counterexamples.npz", fixed_noise=noise.numpy(),
                            law_variance_jacobian=law_variance.numpy(), law_scale_jacobian=law_scale.numpy(),
                            fixed_path_jacobian=path_scale.numpy())
    return record


def plot(records, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    floor = 1e-16
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.7))
    for name, label, color in (("separated", "Separated: (0.2, 0.8)", "#2463a0"),
                               ("delta_0.0001", "Near: (0.5, 0.5001)", "#c37817"),
                               ("delta_0", "Coincident: (0.5, 0.5)", "#328068")):
        s = np.asarray(records[name]["fixed_scaled"]["singular_values"])
        left.plot(range(1,5), np.maximum(s, floor), "o-", label=label, color=color, lw=2)
    left.set(yscale="log", xlabel="Singular-value index", ylabel="Singular value (fixed moment scaling)",
             title="Three parameter regimes", xticks=[1,2,3,4], ylim=(floor/3, 1e3))
    left.legend(fontsize=9, loc="lower left")
    sweep = np.array([records[f"delta_{d:g}"]["fixed_scaled"]["singular_values"] for d in DELTAS])
    for j, color in enumerate(("#2463a0", "#328068", "#c37817", "#a84b73")):
        right.plot(DELTAS, np.maximum(sweep[:,j]/sweep[:,0], floor), "o-", label=f"s{j+1} / s1", color=color)
    right.set_xscale("symlog", linthresh=1e-6)
    right.set(yscale="log", xlabel=r"Persistence gap $\delta$ (zero shown explicitly)",
              ylabel="Relative singular value", title="Approaching coincident persistence", ylim=(floor/3, 3))
    right.axhline(1e-9, color="#888888", lw=.8, ls="--", label="rtol = 1e-9")
    right.legend(fontsize=8, loc="lower right")
    for ax in (left, right):
        ax.grid(which="major", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Local sensitivity of two independent Gaussian AR(1) components", fontsize=14)
    fig.text(.5, .025, "Raw history T=128; zero starts; v1=v2=1; coordinate scale I; moment scaling fixed at each point.\n"
             "Values below 1e-16 are plotted at the floor. Numerical rank is not a global-identification certificate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.115,1,.94))
    fig.savefig(destination / "sensitivity_spectrum.png", dpi=180)
    fig.savefig(destination / "sensitivity_spectrum.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--worker", choices=tuple(CASES))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        run_case(args.worker, output)
        return
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "PYTHONDONTWRITEBYTECODE":"1",
                   "MPLCONFIGDIR": str(output / "matplotlib")}
    os.environ["MPLCONFIGDIR"] = environment["MPLCONFIGDIR"]
    for name in CASES:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name, "--output", str(output)],
                       cwd=ROOT, env=environment, check=True)
    setup_threads()
    records = {name: json.loads((output/f"{name}.json").read_text()) for name in CASES}
    examples = counterexamples(output)
    manifest = {"schema":SCHEMA, "source_revision":subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(),
                "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "delta_sweep":DELTAS, "records":records, "counterexamples":examples,
                "execution":"fresh sequential subprocess per point; two intra-op CPU threads; no training; no derivative cache"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    figures = args.figures.resolve() if args.figures else output / "figures"
    if not figures.is_relative_to(ROOT):
        parser.error("figures must stay inside this research checkout")
    plot(records, figures)
    for name in ("separated", "delta_0.0001", "delta_0"):
        r=records[name]
        print(name, "scaled spectrum", r["fixed_scaled"]["singular_values"], "ranks", r["fixed_scaled"]["rank_by_rtol"])
    print("Artifacts:", output, "Figures:", figures)


if __name__ == "__main__":
    main()
