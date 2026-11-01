"""Independent likelihood diagnostics for fractional memory, on CPU.

Whittle is an approximate frequency-domain likelihood; it does not certify
stationary initialization. No neural training.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from scipy.optimize import minimize_scalar
import torch
import timegram as tg

torch.set_num_threads(2)


def fractional_recovery(d, seed, nobs=16384, prehistory=8192):
    start=time.perf_counter()
    model=tg.fractional_integrate(d,presample=prehistory)
    y=model.simulate(n=1,length=nobs,seed=seed,backend="fast",device="cpu").context[0]
    y=y-y.mean()
    frequencies=2*np.pi*np.fft.rfftfreq(nobs)[1:-1]
    periodogram=np.abs(np.fft.rfft(y)[1:-1])**2/nobs
    log_difference=np.log(2*np.sin(frequencies/2))
    def objective(order):
        log_shape=-2*order*log_difference
        variance=np.mean(periodogram*np.exp(-log_shape))
        return len(log_shape)*np.log(variance)+log_shape.sum()
    fit=minimize_scalar(objective,bounds=(-0.45,0.45),method="bounded")
    estimated=float(fit.x)
    variance=float(np.mean(periodogram*np.exp(2*estimated*log_difference)))
    error=abs(estimated-d)
    return {"case":"fractional_noise","seed":seed,"nobs":nobs,"prehistory":prehistory,
            "method":"profile Whittle approximation","truth_d":d,"estimated_d":estimated,
            "absolute_error_d":error,"estimated_innovation_variance":variance,"tolerance_d":0.06,
            "passed":bool(fit.success and error<0.06 and abs(variance-1)<0.12),
            "limitation":"Finite Gaussian prehistory; neither exact stationary initialization nor exact likelihood",
            "seconds":time.perf_counter()-start}



def main():
    rows=[fractional_recovery(d,seed) for d in [0.,.2,.35] for seed in [11,29]]
    output=Path("output/fractional_recovery.json")
    output.write_text(json.dumps({"device":"cpu","threads":2,"results":rows},indent=2),encoding="utf-8")
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} fractional recovery runs passed")
    return int(not all(r['passed'] for r in rows))

if __name__=="__main__":
    raise SystemExit(main())
