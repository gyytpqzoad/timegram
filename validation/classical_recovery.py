"""Long-series CPU recovery by independent classical estimators.

Generators use only grammar sources and atomic filters, never tg.ets/arima.
Declared fixed tolerances are diagnostic alarms, not confidence intervals.
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
import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import torch
import timegram as tg
from timegram.blocks import damped_trend

torch.set_num_threads(2)


def arima_cases():
    def chain(phi=(), theta=(), d=0, seasonal_phi=(), seasonal_theta=(), D=0, period=1, shape=None, scale=1):
        m = tg.noise(tg.normal() if shape is None else shape, scale=scale)
        if theta: m = m >> tg.ma(len(theta), theta=list(theta))
        if seasonal_theta: m = m >> tg.ma(len(seasonal_theta), period=period, theta=list(seasonal_theta))
        if phi: m = m >> tg.ar(len(phi), phi=list(phi))
        if seasonal_phi: m = m >> tg.ar(len(seasonal_phi), period=period, phi=list(seasonal_phi))
        if d: m = m >> tg.integrate(d)
        if D: m = m >> tg.integrate(D, period=period)
        return m
    specifications = [
        ("AR1", dict(phi=(0.65,), scale=1.2), (1,0,0), (0,0,0,0), {"ar.L1":0.65,"sigma2":1.44}),
        ("MA2", dict(theta=(0.45,-0.2)), (0,0,2), (0,0,0,0), {"ma.L1":0.45,"ma.L2":-0.2,"sigma2":1}),
        ("ARMA21", dict(phi=(0.55,-0.2),theta=(0.4,)), (2,0,1), (0,0,0,0),
         {"ar.L1":0.55,"ar.L2":-0.2,"ma.L1":0.4,"sigma2":1}),
        ("ARIMA111", dict(phi=(0.55,),theta=(0.35,),d=1), (1,1,1), (0,0,0,0),
         {"ar.L1":0.55,"ma.L1":0.35,"sigma2":1}),
        ("SARMA_4",dict(phi=(0.35,),theta=(0.3,),seasonal_phi=(0.55,),seasonal_theta=(-0.25,),period=4),
         (1,0,1),(1,0,1,4),{"ar.L1":0.35,"ma.L1":0.3,"ar.S.L4":0.55,"ma.S.L4":-0.25,"sigma2":1}),
        ("SARIMA_011_011_4",dict(theta=(-0.4,),d=1,seasonal_theta=(-0.35,),D=1,period=4),
         (0,1,1),(0,1,1,4),{"ma.L1":-0.4,"ma.S.L4":-0.35,"sigma2":1}),
        ("AR1_Student_QMLE",dict(phi=(0.6,),shape=tg.student_t(8),scale=np.sqrt(6/8)),
         (1,0,0),(0,0,0,0),{"ar.L1":0.6,"sigma2":1}),
        ("AR1_mixture_QMLE",dict(phi=(0.6,),shape=tg.mixture(tg.normal(),tg.normal(0,5),weights=[0.97,0.03]),
                                   scale=1/np.sqrt(0.97+0.03*25)),
         (1,0,0),(0,0,0,0),{"ar.L1":0.6,"sigma2":1}),
    ]
    result = [{"name":n,"model":chain(**s),"order":o,"seasonal_order":so,"truth":t,"kind":"sarimax",
               "likelihood":"Gaussian QMLE" if "QMLE" in n else "Gaussian MLE"} for n,s,o,so,t in specifications]
    w, sa, sb = 0.35, 1.2, 0.7
    a = tg.noise(scale=sa) >> tg.ar(1,phi=[0.6])
    b = tg.noise(scale=sb) >> tg.ar(1,phi=[0.6])
    e = tg.noise()
    shared = tg.mix((sa*e)>>tg.ar(1,phi=[0.6]),(sb*e)>>tg.ar(1,phi=[0.6]),weight=w)
    for n,m,v in [("convex_independent_AR1",tg.mix(a,b,weight=w),w*w*sa*sa+(1-w)**2*sb*sb),
                  ("convex_shared_AR1",shared,(w*sa+(1-w)*sb)**2)]:
        result.append({"name":n,"model":m,"order":(1,0,0),"seasonal_order":(0,0,0,0),
                       "truth":{"ar.L1":0.6,"sigma2":v},"kind":"sarimax","likelihood":"Gaussian MLE"})
    return result


def ets_cases():
    result = []
    for name,trend,seasonal,damping in [("SES",False,False,None),("Holt",True,False,None),
            ("Holt_Winters",True,True,None),("Damped_Holt",True,False,0.9),("Damped_Holt_Winters",True,True,0.9)]:
        alpha,beta,gamma,level,slope,period = 0.35,0.06,0.15,2.0,0.03,4
        season = [1.0,-0.5,0.25,-0.75]
        e = tg.noise()
        m = tg.constant(level)+e+alpha*(e>>tg.integrate()>>tg.lag(1))
        truth = {"smoothing_level":alpha}
        if trend and damping is None:
            m = m+tg.trend(slope)+beta*(e>>tg.integrate(2)>>tg.lag(1))
            truth["smoothing_trend"] = beta
        elif damping is not None:
            m = m+damped_trend(slope,damping)+(beta*damping)*(e>>tg.ar(1,phi=[damping],init="zero")>>tg.integrate()>>tg.lag(1))
            truth.update(smoothing_trend=beta,damping_trend=damping)
        if seasonal:
            m = m+tg.pattern(period,values=season)+gamma*(e>>tg.integrate(period=period)>>tg.lag(period))
            truth["smoothing_seasonal"] = gamma
        truth["sigma2"] = 1.0
        result.append({"name":name,"model":m,"kind":"ets","truth":truth,"trend":trend,"seasonal":seasonal,
                       "damping":damping,"level":level,"slope":slope,"season":season,"period":period,
                       "likelihood":"Gaussian MLE with known initial states"})
    return result


def all_cases():
    return arima_cases()+ets_cases()


def fit_case(case,y):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        if case["kind"] == "sarimax":
            model = SARIMAX(y,order=case["order"],seasonal_order=case["seasonal_order"],trend="n",
                            simple_differencing=True,enforce_stationarity=True,enforce_invertibility=True)
            fit = model.fit(disp=False,maxiter=300,cov_type="robust")
            estimates = dict(zip(fit.param_names,map(float,fit.params)))
        else:
            model = ETSModel(y,error="add",trend="add" if case["trend"] else None,
                seasonal="add" if case["seasonal"] else None,damped_trend=case["damping"] is not None,
                seasonal_periods=case["period"] if case["seasonal"] else None,
                initialization_method="known",initial_level=case["level"],
                initial_trend=case["slope"] if case["trend"] else None,
                initial_seasonal=case["season"] if case["seasonal"] else None)
            fit = model.fit(disp=False,maxiter=500)
            estimates = dict(zip(fit.param_names,map(float,fit.params)))
            estimates["sigma2"] = float(fit.scale)
        captured = [str(w.message) for w in messages]
    return estimates,bool(fit.mle_retvals.get("converged",False)),captured


def recovery(case,nobs,seed):
    started = time.perf_counter()
    y = case["model"].simulate(n=1,length=nobs,seed=seed).context[0]
    estimates,converged,messages = fit_case(case,y)
    checks = {}
    for name,truth in case["truth"].items():
        tolerance = (0.12 if "QMLE" in case["name"] else 0.08)*max(1.0,truth) if name == "sigma2" else 0.065
        # A zero estimate must not pass for the true 0.06 trend coefficient.
        # Tightened after the initial smoke run; seed 47 is an additional check.
        if name == "smoothing_trend": tolerance = 0.02
        if name == "damping_trend": tolerance = 0.035
        error = abs(estimates[name]-truth)
        checks[name] = {"truth":truth,"estimate":estimates[name],"absolute_error":error,
                        "tolerance":tolerance,"passed":error<tolerance}
    return {"case":case["name"],"nobs":nobs,"seed":seed,"backend":"reference","likelihood":case["likelihood"],
            "converged":converged,"checks":checks,"passed":converged and all(c["passed"] for c in checks.values()),
            "warnings":messages,"seconds":time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nobs",type=int,default=12000)
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,29,47])
    parser.add_argument("--cases",nargs="*")
    parser.add_argument("--output",type=Path,default=Path("output/classical_recovery.json"))
    args = parser.parse_args()
    rows = []
    for case in [c for c in all_cases() if not args.cases or c["name"] in args.cases]:
        for seed in args.seeds:
            try: row = recovery(case,args.nobs,seed)
            except Exception as exc: row = {"case":case["name"],"seed":seed,"passed":False,"error":f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(json.dumps(row),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({"environment":{"python":platform.python_version(),"numpy":np.__version__,
                "scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"torch":torch.__version__,"device":"cpu","threads":2},
                "results":rows},indent=2),encoding="utf-8")
    failed = [r for r in rows if not r["passed"]]
    print(f"{len(rows)-len(failed)}/{len(rows)} recovery runs passed",flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
