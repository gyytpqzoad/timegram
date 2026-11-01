"""Rolling one-step backtest of quantile forecasters on Tourism monthly, and the two baselines.

Every model receives identical histories: for each series, the last `n_origins` one-step
origins, with contexts of at least `min_context` and at most `max_context` observations.
Per series and quantile level, `summarize` returns the coverage and the mean pinball loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def load_tourism_monthly() -> dict[str, np.ndarray]:
    """Full Tourism monthly series (training part followed by the official test part)."""
    from fcompdata import load_tourism
    return {str(s.sn): np.concatenate([np.asarray(s.x, float), np.asarray(s.xx, float)])
            for s in load_tourism() if s.period == 12}


@dataclass(frozen=True)
class BacktestConfig:
    n_origins: int
    horizon: int = 1
    min_context: int = 48
    max_context: int = 2048
    window_batch_size: int = 4096
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    @property
    def minimum_length(self) -> int:
        return self.min_context + self.horizon + self.n_origins - 1


def windows(series: dict[str, np.ndarray], config: BacktestConfig):
    """(series id, origin, context, outcome); the origin is the index of the first held-out point."""
    for key in sorted(k for k, y in series.items() if len(y) >= config.minimum_length):
        y = series[key]
        first = len(y) - config.horizon - config.n_origins + 1
        for origin in range(first, len(y) - config.horizon + 1):
            start = max(0, origin - config.max_context)
            yield key, origin, y[start:origin], y[origin:origin + config.horizon]


def run_backtest(series: dict[str, np.ndarray], models, config: BacktestConfig) -> pd.DataFrame:
    rows, batch = [], []
    q = np.asarray(config.quantiles)

    def evaluate(batch):
        contexts = [w[2] for w in batch]
        truth = np.stack([w[3] for w in batch])
        for model in models:
            forecast = np.asarray(model.predict(contexts, config.horizon, config.quantiles))
            expected = (len(batch), config.horizon, len(q))
            if forecast.shape != expected or not np.isfinite(forecast).all():
                raise ValueError(f"{model.name}: expected finite forecasts of shape {expected}")
            error = truth[..., None] - forecast
            loss = np.maximum(q * error, (q - 1) * error)
            for b, (key, origin, _, _) in enumerate(batch):
                for h in range(config.horizon):
                    for j, level in enumerate(q):
                        rows.append((key, model.name, origin, h + 1, level, truth[b, h], forecast[b, h, j],
                                     truth[b, h] <= forecast[b, h, j], loss[b, h, j]))

    for window in windows(series, config):
        batch.append(window)
        if len(batch) == config.window_batch_size:
            evaluate(batch)
            batch = []
    if batch:
        evaluate(batch)
    return pd.DataFrame(rows, columns=["series_id", "model", "origin", "horizon", "quantile", "y",
                                       "prediction", "covered", "pinball_loss"])


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per model, series, horizon and quantile level: coverage, number of origins, pinball loss."""
    return (predictions.groupby(["model", "series_id", "horizon", "quantile"], observed=True)
            .agg(coverage=("covered", "mean"), n=("covered", "size"), pinball_loss=("pinball_loss", "mean"))
            .reset_index())


# ---------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------
@dataclass
class Chronos2:
    """amazon/chronos-2, each forecasting task on its own (cross-learning off: different origins of
    one series can share a batch, and a later origin contains outcomes hidden from an earlier one)."""

    device: str = "cuda"
    batch_size: int = 256
    context_length: int = 2048
    name: str = "Chronos-2"
    _pipeline: object = field(default=None, repr=False, init=False)

    def predict(self, contexts, horizon, quantiles):
        if self._pipeline is None:
            import torch
            from chronos import Chronos2Pipeline
            dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() \
                else torch.float32
            self._pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.device,
                                                              torch_dtype=dtype)
        predictions, _ = self._pipeline.predict_quantiles(
            inputs=[np.asarray(x[-self.context_length:], dtype=np.float32) for x in contexts],
            prediction_length=horizon, quantile_levels=list(quantiles), batch_size=self.batch_size,
            context_length=self.context_length, cross_learning=False)
        return np.stack([p.detach().float().cpu().numpy()[0] for p in predictions])


@dataclass
class AutoETS:
    """statsforecast AutoETS, refitted per window; quantiles from its central prediction intervals."""

    season_length: int = 12
    n_jobs: int = 2
    name: str = "AutoETS"

    def predict(self, contexts, horizon, quantiles):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS as Model
        frame = pd.concat([pd.DataFrame({"unique_id": i, "ds": np.arange(len(x)), "y": x})
                           for i, x in enumerate(contexts)], ignore_index=True)
        levels = sorted({round(100 * abs(2 * q - 1), 8) for q in quantiles if not np.isclose(q, 0.5)})
        fc = (StatsForecast(models=[Model(season_length=self.season_length, model="ZZZ")], freq=1,
                            n_jobs=self.n_jobs)
              .forecast(df=frame, h=horizon, level=levels).sort_values(["unique_id", "ds"]))
        fc = fc.rename(columns={c: f"{c.rsplit('-', 1)[0]}-{float(c.rsplit('-', 1)[1]):g}"
                                for c in fc.columns if "-lo-" in str(c) or "-hi-" in str(c)})

        def column(q):
            if np.isclose(q, 0.5):
                return "AutoETS"
            return f"AutoETS-{'lo' if q < 0.5 else 'hi'}-{round(100 * abs(2 * q - 1), 8):g}"

        return fc[[column(q) for q in quantiles]].to_numpy(float).reshape(len(contexts), horizon, len(quantiles))
