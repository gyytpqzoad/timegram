"""timegram: a grammar for structural time series models, solved by Neural Synthetic
Pre-training.

    import timegram as tg
    model = tg.ets("AAA", period=4)
    estimator = model.estimate(tg.forecast(h=8)).pretrain()
    result = estimator(series)
"""
from . import identity, priors, worlds
from .blocks import (ar, constant, counts, damped_trend, difference, exp, indicator, integrate, lag, ma, noise, normalize,
                     pattern, trend)
from .compounds import arima, ets, local_level, local_trend, seasonal
from .families import arfima
from .fractional import fractional_difference, fractional_integrate, fractional_weights
from .graph import model_graph, plot_graph
from .volatility import arch, garch
from .dists import (bernoulli, choice, dirichlet, integers, invertible, l1_ball, laplace, mixture, normal, stationary, student_t,
                    uniform)
from . import library
from .library import pretrained
from .model import Block, add, at_least, compose, convex, feed, mix, product, scale, select
from .moves import moves
from .complexity import complexity
from .witnesses import witnesses
from .nets import ResMLP
from . import objective
from .objective import bias, mean, miscoverage, risk, worst
from .optim import AdamW, MuonAdamW
from .problem import absolute, forecast, missing, params, pinball, squared, which
from .simulate import simulate
from .train import TrainConfig

__all__ = [
    "damped_trend", "dirichlet", "exp", "invertible", "normalize", "model_graph", "plot_graph",
    "arch", "arfima", "fractional_difference", "fractional_integrate", "fractional_weights", "garch",
    "absolute", "AdamW", "complexity", "forecast", "identity", "library", "missing", "moves", "MuonAdamW", "params", "pretrained", "witnesses",
    "ResMLP", "squared", "TrainConfig",
    "bias", "mean", "miscoverage", "objective", "pinball", "risk", "worst", "which",
    "ar", "arima", "add", "Block", "choice", "constant", "compose", "difference", "ets", "feed",
    "bernoulli", "counts", "indicator", "integers", "integrate", "l1_ball", "lag", "laplace", "local_level", "local_trend", "ma",
    "at_least", "convex", "mix", "mixture", "noise", "product", "normal", "pattern", "priors", "worlds", "scale", "seasonal", "select",
    "simulate", "stationary", "student_t", "trend", "uniform",
]
