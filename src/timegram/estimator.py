"""A trained estimator: apply it to a series, get a Result."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .train import normalize


@dataclass
class Result:
    """Estimates for one series. Point estimates for now (the problems in this release are
    trained with point losses); `forecast` and `params` are DataFrames."""

    forecast: pd.DataFrame | None
    params: pd.DataFrame | None

    def __repr__(self):
        parts = []
        if self.params is not None and len(self.params):
            parts.append(f"params:\n{self.params.to_string(index=False)}")
        if self.forecast is not None:
            parts.append(f"forecast:\n{self.forecast.to_string(index=False)}")
        return "\n\n".join(parts)


class Estimator:
    def __init__(self, problem, net, config, horizon: int, slots: dict, history: list, key: str | None = None):
        self.problem, self.net, self.config = problem, net, config
        self.horizon, self.slots, self.history, self.key = horizon, slots, history, key
        self.context = config.lengths[1]

    def to(self, device):
        self.net.to(device)
        return self

    @torch.no_grad()
    def predict(self, histories, batch_size: int = 8192) -> np.ndarray:
        """Raw network output for many series: [n, horizon + parameter slots], forecasts
        de-normalized to each series' scale. Each history uses its last `context` points."""
        device = next(self.net.parameters()).device
        L, out = self.context, []
        for i in range(0, len(histories), batch_size):
            chunk = histories[i:i + batch_size]
            ctx = np.full((len(chunk), L), np.nan, dtype=np.float32)
            for j, h in enumerate(chunk):
                h = np.asarray(h, dtype=np.float32)[-L:]
                ctx[j, L - len(h):] = h
            x, m, loc, scale = normalize(torch.as_tensor(ctx, device=device))
            y = self.net(x, m)
            if self.horizon:
                f = self.problem.forecast_estimand()
                q = f.decode(y[:, :f.width]) * scale.unsqueeze(-1) + loc.unsqueeze(-1)
                y[:, :f.width] = q.flatten(1)
            for path, K in self.problem.categorical().items():          # class probabilities
                start = self.slots[path][0]
                y[:, start:start + K] = torch.softmax(y[:, start:start + K], 1)
            out.append(y.cpu().numpy())
        return np.concatenate(out).astype(float)

    @property
    def quantiles(self) -> tuple[float, ...]:
        f = self.problem.forecast_estimand()
        return f.quantiles if f else ()

    def forecast(self, histories) -> np.ndarray:
        """Forecasts: points [n, horizon], or quantiles [n, horizon, levels]."""
        Q = len(self.quantiles)
        y = self.predict(histories)[:, :self.horizon * max(Q, 1)]
        return y.reshape(len(y), self.horizon, Q) if Q else y

    def __call__(self, series) -> Result:
        values = np.asarray(series, dtype=float)
        y = self.predict([values])[0]
        fc = None
        if self.horizon:
            fc = pd.DataFrame({"period": _future_index(series, self.horizon)})
            if self.quantiles:
                q = y[:self.horizon * len(self.quantiles)].reshape(self.horizon, -1)
                for i, level in enumerate(self.quantiles):
                    fc[f"q{level:g}"] = q[:, i]
            else:
                fc["estimate"] = y[:self.horizon]
        rows = []
        for path, (start, size) in self.slots.items():
            v = y[start:start + size]
            rows.append({"parameter": path, "estimate": float(v[0]) if size == 1 else np.round(v, 4)})
        return Result(fc, pd.DataFrame(rows) if rows else None)

    def __repr__(self):
        return f"Estimator({self.key or 'unsaved'})\n{self.problem.describe()}"


def _future_index(series, h):
    idx = getattr(series, "index", None)
    if isinstance(idx, pd.PeriodIndex):
        return [str(p) for p in pd.period_range(idx[-1] + 1, periods=h, freq=idx.freq)]
    if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
        return list(pd.date_range(idx[-1], periods=h + 1, freq=idx.freq)[1:])
    n = len(series)
    return list(range(n + 1, n + h + 1))
