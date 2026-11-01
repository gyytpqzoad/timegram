"""CPU recovery of grammar-generated volatility using independent arch MLE.

Protocol fixed before the first fit: 30,000 observations, burn-in 2,000,
seeds 11/29/47, package-default starts (no true parameters passed to fit).
Pointwise independent tests, rather than MLE alone, verify the recursions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import arch as arch_package
from arch.univariate import ARX, ZeroMean, GARCH as ArchGARCH, HARCH, Normal, StudentsT
import numpy as np
import pandas as pd
import scipy
import torch
import timegram as tg

torch.set_num_threads(2)

BURN_IN = 2000
NOBS = 30000
SEEDS = [11, 29, 47]
# Absolute diagnostic tolerances, not confidence intervals. Every nonzero truth
# exceeds its tolerance, so an omitted term cannot pass by estimating zero.
TOLERANCES = {"omega": .04, "alpha": .035, "beta": .055, "gamma": .045,
              "ar": .025, "Const": .05, "nu": 1.5}


def cases():
    specs = [
        dict(name="ARCH1", p=0, q=1, omega=.7, alpha=[.3], beta=[]),
        dict(name="ARCH3", p=0, q=3, omega=.4, alpha=[.3, .2, .1], beta=[]),
        dict(name="GARCH11", p=1, q=1, omega=.1, alpha=[.18], beta=[.72]),
        dict(name="GARCH_variance2_shocks1", p=2, q=1, omega=.12, alpha=[.2], beta=[.48, .2]),
        dict(name="GARCH_variance1_shocks2", p=1, q=2, omega=.12, alpha=[.15, .1], beta=[.63]),
        dict(name="AR1_GARCH", p=1, q=1, omega=.1, alpha=[.18], beta=[.72], phi=[.55], mean=1.5),
        dict(name="AR2_GARCH", p=1, q=1, omega=.1, alpha=[.18], beta=[.72], phi=[.45, -.2], mean=1.2),
        dict(name="GJR_GARCH", p=1, q=1, omega=.1, alpha=[.1], beta=[.72], gamma=[.16]),
        dict(name="AR1_GJR_GARCH", p=1, q=1, omega=.1, alpha=[.1], beta=[.72], gamma=[.16],
             phi=[-.35], mean=.4),
        dict(name="TARCH", p=1, q=1, omega=.15, alpha=[.15], beta=[.65], gamma=[.12], power=1.),
        dict(name="AVGARCH", p=1, q=1, omega=.12, alpha=[.2], beta=[.7], power=1.),
        dict(name="Student_GARCH", p=1, q=1, omega=.1, alpha=[.18], beta=[.72], nu=8.),
        dict(name="Student_AR1_GARCH", p=1, q=1, omega=.1, alpha=[.18], beta=[.72], nu=8.,
             phi=[.55], mean=1.5),
    ]
    result = []
    for spec in specs:
        nu = spec.get("nu")
        shape = tg.normal() if nu is None else tg.student_t(nu)
        scale = 1. if nu is None else np.sqrt((nu-2)/nu)
        source = tg.noise(shape, scale=scale)
        filt = tg.garch(spec["p"], spec["q"], omega=spec["omega"], alpha=spec["alpha"],
            beta=spec["beta"], gamma=spec.get("gamma"), power=spec.get("power", 2.), burn_in=BURN_IN)
        model = source >> filt
        phi = spec.get("phi", [])
        if phi:
            model = tg.constant(spec["mean"]) + (model >> tg.ar(len(phi), phi=phi))
        truth = {"omega": spec["omega"]}
        for key in ("alpha", "gamma", "beta"):
            truth.update({f"{key}[{j+1}]": v for j, v in enumerate(spec.get(key, []))})
        if nu is not None: truth["nu"] = nu
        if phi:
            truth["Const"] = spec["mean"] * (1-sum(phi))
            truth.update({f"y[{j+1}]": v for j, v in enumerate(phi)})
        result.append({**spec, "model": model, "truth": truth, "fit_kind": "garch"})

    # HARCH(1,5,22) is a constrained ARCH(22), not a new simulation primitive.
    horizons, weights = [1, 5, 22], [.15, .25, .3]
    alpha = np.zeros(max(horizons))
    for horizon, weight in zip(horizons, weights):
        alpha[:horizon] += weight / horizon
    result.append({"name": "HARCH_1_5_22", "model": tg.noise() >> tg.arch(22,
        omega=.3, alpha=alpha, burn_in=BURN_IN), "fit_kind": "harch", "horizons": horizons,
        "truth": {"omega": .3, **{f"alpha[{h}]": w for h, w in zip(horizons, weights)}}})
    return result


def fit_case(case, values):
    y = pd.Series(values, name="y")
    distribution = Normal() if "nu" not in case else StudentsT()
    if case["fit_kind"] == "harch":
        volatility = HARCH(lags=case["horizons"])
    else:
        # arch uses p=shock lags, q=variance lags; timegram uses the reverse.
        volatility = ArchGARCH(p=case["q"], o=case["q"] if "gamma" in case else 0,
                              q=case["p"], power=case.get("power", 2.))
    if case.get("phi"):
        estimator = ARX(y, lags=len(case["phi"]), constant=True, volatility=volatility,
                        distribution=distribution, rescale=False)
    else:
        estimator = ZeroMean(y, volatility=volatility, distribution=distribution, rescale=False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted = estimator.fit(disp="off", show_warning=True, options={"maxiter": 1000, "ftol": 1e-9})
    estimates = {k: float(v) for k, v in fitted.params.items()}
    optimizer = fitted.optimization_result
    return estimates, int(fitted.convergence_flag), str(optimizer.message), [str(w.message) for w in captured]


def tolerance(name):
    if name.startswith("y["): return TOLERANCES["ar"]
    return TOLERANCES[name.split("[")[0]]


def recovery(case, nobs=NOBS, seed=11):
    started = time.perf_counter()
    y = case["model"].simulate(n=1, length=nobs, seed=seed).context[0]
    estimates, flag, message, messages = fit_case(case, y)
    checks = {}
    for name, truth in case["truth"].items():
        error, limit = abs(estimates[name]-truth), tolerance(name)
        checks[name] = {"truth": truth, "estimate": estimates[name], "absolute_error": error,
                        "tolerance": limit, "passed": bool(error < limit)}
    return {"case": case["name"], "nobs": nobs, "seed": seed, "burn_in": BURN_IN,
            "simulator": "timegram atomic grammar", "backend": "reference", "estimator": "arch MLE",
            "true_parameters_used_to_initialize_fit": False, "convergence_flag": flag,
            "optimizer_message": message, "warnings": messages, "checks": checks,
            "passed": flag == 0 and all(c["passed"] for c in checks.values()),
            "seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nobs", type=int, default=NOBS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("output/volatility_recovery.json"))
    args = parser.parse_args()
    all_cases = cases()
    if args.cases and set(args.cases)-{c["name"] for c in all_cases}:
        parser.error("unknown case name")
    selected = [c for c in all_cases if not args.cases or c["name"] in args.cases]
    rows = []
    report = {"environment": {"arch": arch_package.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "device": "cpu", "threads": 2},
        "protocol": {"burn_in": BURN_IN, "nobs": args.nobs, "seeds": args.seeds, "tolerances": TOLERANCES,
                     "initialization": "finite burn-in; fit uses package-default starts and backcast"}, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in selected:
        for seed in args.seeds:
            try:
                row = recovery(case, args.nobs, seed)
            except Exception as exc:
                row = {"case": case["name"], "seed": seed, "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    failed = sum(not row["passed"] for row in rows)
    print(f"{len(rows)-failed}/{len(rows)} volatility recovery runs passed", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
