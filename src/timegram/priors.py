"""Joint priors: the low-informative ARIMA/SARIMA "tree" prior (`SarimaTree`) and the
explicit starting states of its integrators. Ready-made worlds built from them are in
`tg.worlds`.

`SarimaTree` is a joint distribution over an ARIMA's structure and coefficients. Every
decision is a uniform draw over its options, with auto.arima's structural rules as
constraints:

    d           uniform over d_choices
    seasonal?   with probability seasonal_prob; if so
        D       1 with probability D1_prob, else 0
        family  uniform over {SAR, SMA, SARMA}, plus "no seasonal polynomial" when D = 1
        P, Q    uniform on 1..max_seasonal_order
    family      white noise / AR / MA / ARMA (family_probs), orders uniform on 1..max_order
    shape       order >= 2: only the last lag w.p. sparse_prob, else dense
    region      each polynomial: L1 ball w.p. l1_prob, else uniform partial autocorrelations
    rules       d + D <= 2; a constant only where d + D <= 1, nonzero w.p. const_prob,
                c ~ U(-const_max, const_max) inside the recursion
    states      integrators start at the first observation; their starting states are
                explicit: a zero-sum pattern season0 after the seasonal
                integrator (D = 1) and a level0 between the integrators (d = D = 1). They are
                drawn as the state the original sampler's random warm-up of k ~ U(0, warm_max
                + 5 s) steps would accumulate, from the series' own spectrum (`starting_states`),
                without simulating the warm-up.

One difference from the original sampler: it drew the length first and then a structure
identifiable at that length; here the structure is drawn first and the length after,
uniform over the lengths that identify it (the grammar's rule for every model).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dists import Joint, pacf_to_coefs, pacf_to_coefs_batch


@dataclass(eq=False)
class SarimaTree(Joint):

    def own_support(self, context):
        entries = {key: {"kind": "stationary" if key in ("phi", "Phi") else "invertible"}
                   for key in ("phi", "theta", "Phi", "Theta")}
        entries.update({key: {"kind": "integer", "integer": True, "interval": interval}
                        for key, interval in {"p": (0, self.max_order), "q": (0, self.max_order),
                                              "P": (0, self.max_seasonal_order), "Q": (0, self.max_seasonal_order),
                                              "d": (min(self.d_choices), max(self.d_choices)), "D": (0, 1)}.items()})
        d = np.asarray(self.d_choices)
        entries["d"]["integer"] = bool(d.dtype.kind in "iuf" and np.isfinite(d).all()
                                        and (d == np.floor(d)).all())
        for info in entries.values():
            info["finite"] = True
        return context.joint(entries)

    period: int = 4
    d_choices: tuple[int, ...] = (0, 1)
    max_order: int = 3
    family_probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    seasonal_prob: float = 0.5
    D1_prob: float = 0.5
    max_seasonal_order: int = 2
    sparse_prob: float = 0.5
    l1_prob: float = 0.5
    pacf_max: float = 0.95
    const_prob: float = 0.5
    const_max: float = 1.0
    warm_max: int = 40

    keys = ("p", "q", "d", "P", "Q", "D", "phi", "theta", "Phi", "Theta", "mean", "level0", "season0")

    def shape_for(self, key: str) -> tuple[int, ...]:
        """Coefficient vectors include zero-valued inactive lags up to the maximum
        order. Estimating them targets these padded coefficients, not latent orders."""
        if key in ("phi", "theta"):
            return (self.max_order,)
        if key in ("Phi", "Theta"):
            return (self.max_seasonal_order,)
        if key == "season0":
            return (self.period,)
        return ()

    def describe(self):
        return f"sarima_tree(period={self.period})"

    def may_be_zero(self, key: str) -> bool:
        """Whether a draw of `key` can be exactly zero (a fixed, known level)."""
        if key == "mean":
            return self.const_prob < 1 or max(self.d_choices) + (self.period > 1 and self.seasonal_prob > 0) > 1
        return key in ("level0", "season0")

    # -- reference: one series -----------------------------------------------------------
    def sample(self, rng, draw):
        s = self.period
        for _ in range(500):
            d = self.d_choices[rng.integers(len(self.d_choices))]
            p, q = self._part(rng, rng.choice(4, p=self._fam()), self.max_order)
            P = Q = D = 0
            if s > 1 and rng.random() < self.seasonal_prob:
                D = int(rng.random() < self.D1_prob)
                fams = (1, 2, 3) if D == 0 else (0, 1, 2, 3)        # "none" needs D = 1
                P, Q = self._part(rng, fams[rng.integers(len(fams))], self.max_seasonal_order)
            if d + D <= 2:
                break
        # Coefficients are drawn in the AR convention 1 - sum c B^j (stationary /
        # invertible by construction); timegram's MA is y = x + sum theta x, so theta = -c.
        phi, theta = self._coefs(rng, p, self.max_order), -self._coefs(rng, q, self.max_order)
        Phi = self._coefs(rng, P, self.max_seasonal_order)
        Theta = -self._coefs(rng, Q, self.max_seasonal_order)
        c = rng.uniform(-self.const_max, self.const_max) \
            if d + D <= 1 and rng.random() < self.const_prob else 0.0
        mean = c / ((1 - phi.sum()) * (1 - Phi.sum()))          # c / (phi(1) Phi(1))
        level0, season0 = starting_states(rng, phi, theta, Phi, Theta, s, d, D, self.warm_max)
        return dict(p=p, q=q, d=d, P=P, Q=Q, D=D, phi=phi, theta=theta, Phi=Phi, Theta=Theta,
                    mean=mean, level0=level0, season0=season0)

    def _fam(self):
        f = np.asarray(self.family_probs, float)
        return f / f.sum()

    @staticmethod
    def _part(rng, fam, max_order):
        """(white noise | AR | MA | ARMA) -> (ar order, ma order)."""
        k = lambda: int(rng.integers(1, max_order + 1))              # noqa: E731
        if fam == 0:
            return 0, 0
        if fam == 1:
            return k(), 0
        if fam == 2:
            return 0, k()
        return k(), k()

    def _coefs(self, rng, order, size):
        c = np.zeros(size)
        if order == 0:
            return c
        use_l1 = rng.random() < self.l1_prob
        if order >= 2 and rng.random() < self.sparse_prob:          # only the last lag
            mag = rng.uniform(0, 1) if use_l1 else rng.uniform(0, self.pacf_max)
            c[order - 1] = mag if rng.random() < 0.5 else -mag
        elif use_l1:
            direction = rng.uniform(-1, 1, order)
            c[:order] = direction / np.abs(direction).sum() * rng.uniform()
        else:
            c[:order] = pacf_to_coefs(rng.uniform(-self.pacf_max, self.pacf_max, order))
        return c

    # -- fast: n series at once ----------------------------------------------------------
    def sample_batch(self, gen, n, device, draw):
        s = self.period
        u = lambda *shape: torch.rand(*(shape or (n,)), generator=gen, device=device)   # noqa: E731
        ints = lambda lo, hi: torch.randint(lo, hi + 1, (n,), generator=gen, device=device).float()  # noqa: E731

        def structure():
            d = torch.as_tensor(self.d_choices, device=device, dtype=torch.float32)[
                torch.randint(len(self.d_choices), (n,), generator=gen, device=device)]
            fam = torch.multinomial(torch.as_tensor(self._fam(), dtype=torch.float32, device=device), n,
                                    replacement=True, generator=gen)
            p = ((fam == 1) | (fam == 3)).float() * ints(1, self.max_order)
            q = ((fam == 2) | (fam == 3)).float() * ints(1, self.max_order)
            seas = (u() < self.seasonal_prob) & (s > 1)
            D = (seas & (u() < self.D1_prob)).float()
            # seasonal family: uniform over {SAR, SMA, SARMA} (D = 0) or {none, SAR, SMA, SARMA} (D = 1)
            sf = torch.where(D == 1, ints(0, 3), ints(1, 3))
            P = (seas & ((sf == 1) | (sf == 3))).float() * ints(1, self.max_seasonal_order)
            Q = (seas & ((sf == 2) | (sf == 3))).float() * ints(1, self.max_seasonal_order)
            return d, p, q, D, P, Q

        d, p, q, D, P, Q = structure()
        for _ in range(500):                                        # rule d + D <= 2
            bad = d + D > 2
            if not bad.any():
                break
            new = structure()
            d, p, q, D, P, Q = (torch.where(bad, b_, a_) for a_, b_ in zip((d, p, q, D, P, Q), new))

        phi, theta = self._coefs_batch(gen, p, self.max_order), -self._coefs_batch(gen, q, self.max_order)
        Phi = self._coefs_batch(gen, P, self.max_seasonal_order)
        Theta = -self._coefs_batch(gen, Q, self.max_seasonal_order)       # MA sign: see `sample`
        has_c = (d + D <= 1) & (u() < self.const_prob)
        c = torch.where(has_c, self.const_max * (2 * u() - 1), torch.zeros(n, device=device))
        mean = c / ((1 - phi.sum(1)) * (1 - Phi.sum(1)))
        level0, season0 = starting_states_batch(gen, phi, theta, Phi, Theta, s, d, D, self.warm_max)
        return dict(p=p, q=q, d=d, P=P, Q=Q, D=D, phi=phi, theta=theta, Phi=Phi, Theta=Theta,
                    mean=mean, level0=level0, season0=season0)

    def _coefs_batch(self, gen, order, size):
        n, device = len(order), order.device
        u = lambda *shape: torch.rand(*shape, generator=gen, device=device)   # noqa: E731
        lags = torch.arange(size, device=device).unsqueeze(0)
        active = lags < order.unsqueeze(1)                          # lags 1..order
        use_l1 = u(n) < self.l1_prob
        sparse = (order >= 2) & (u(n) < self.sparse_prob)
        # sparse: only the last lag, +-magnitude
        mag = torch.where(use_l1, u(n), self.pacf_max * u(n)) * torch.where(u(n) < 0.5, 1.0, -1.0)
        last = lags == (order.unsqueeze(1) - 1)
        c_sparse = torch.where(last, mag.unsqueeze(1), torch.zeros(n, size, device=device))
        # dense L1 ball
        direction = (2 * u(n, size) - 1) * active
        c_l1 = direction / direction.abs().sum(1, keepdim=True).clamp_min(1e-12) * u(n, 1)
        # dense, uniform partial autocorrelations (zeros beyond the order keep it exact)
        c_pacf = pacf_to_coefs_batch(self.pacf_max * (2 * u(n, size) - 1) * active)
        c = torch.where(sparse.unsqueeze(1), c_sparse, torch.where(use_l1.unsqueeze(1), c_l1, c_pacf))
        return c * active


# ---------------------------------------------------------------------------------------
# standard deviation of the stationary ARMA part (unit innovations)
# ---------------------------------------------------------------------------------------
def _seasonal(coefs, s):
    out = np.zeros(len(coefs) * s)
    out[s - 1::s] = coefs
    return out


def arma_sd(phi, theta, Phi, Theta, s, terms: int = 4000) -> float:
    """sqrt(sum psi_j^2): the stationary standard deviation of
    (1 - phi(B))(1 - Phi(B^s)) w = (1 + theta(B))(1 + Theta(B^s)) e, unit-variance e
    (timegram's MA sign). The impulse response is truncated at `terms` lags."""
    from scipy.signal import lfilter
    ar = np.convolve(np.r_[1.0, -np.asarray(phi, float)], np.r_[1.0, -_seasonal(np.asarray(Phi, float), s)])
    ma = np.convolve(np.r_[1.0, np.asarray(theta, float)], np.r_[1.0, _seasonal(np.asarray(Theta, float), s)])
    impulse = np.zeros(terms)
    impulse[0] = 1.0
    return float(np.sqrt(np.sum(lfilter(ma, ar, impulse) ** 2)))


def arma_sd_batch(phi, theta, Phi, Theta, s, n_fft: int = 8192) -> torch.Tensor:
    """`arma_sd` for a batch, from the transfer function (impulse response by FFT)."""
    def poly(c, sign, seasonal):
        n, k = c.shape
        out = torch.zeros(n, k * (s if seasonal else 1) + 1, device=c.device)
        out[:, 0] = 1.0
        out[:, (s if seasonal else 1)::(s if seasonal else 1)] = sign * c
        return out
    ar = torch.fft.rfft(poly(phi, -1, False), n_fft) * torch.fft.rfft(poly(Phi, -1, True), n_fft)
    ma = torch.fft.rfft(poly(theta, 1, False), n_fft) * torch.fft.rfft(poly(Theta, 1, True), n_fft)
    psi = torch.fft.irfft(ma / ar, n_fft)
    return psi.pow(2).sum(1).sqrt()


# ---------------------------------------------------------------------------------------
# explicit starting states of the integrators
# ---------------------------------------------------------------------------------------
def transfer(phi, theta, Phi, Theta, s, omegas):
    """H(e^{-iw}) = (1 + theta(B))(1 + Theta(B^s)) / ((1 - phi(B))(1 - Phi(B^s))), one
    series (timegram's MA sign), at the frequencies `omegas`."""
    z = np.exp(-1j * np.asarray(omegas, float))[:, None]
    def poly(c, sign, step):
        c = np.asarray(c, float)
        return 1 + sign * (c * z ** (step * np.arange(1, len(c) + 1))).sum(1)
    return poly(theta, 1, 1) * poly(Theta, 1, s) / (poly(phi, -1, 1) * poly(Phi, -1, s))


def transfer_batch(phi, theta, Phi, Theta, s, omegas):
    """`transfer` for a batch: coefficient tensors [n, k] -> [n, len(omegas)] complex."""
    z = torch.exp(-1j * torch.as_tensor(omegas, dtype=torch.float64, device=phi.device))[None, :, None]
    def poly(c, sign, step):
        lags = step * torch.arange(1, c.shape[1] + 1, dtype=torch.float64, device=c.device)
        return 1 + sign * (c.double()[:, None, :] * z ** lags).sum(2)
    return poly(theta, 1, 1) * poly(Theta, 1, s) / (poly(phi, -1, 1) * poly(Phi, -1, s))


def starting_states(rng, phi, theta, Phi, Theta, s, d, D, warm_max):
    """(level0, season0) for one series: the state a warm-up of k ~ U(0, warm_max + 5 s)
    steps would leave in the integrators, drawn from its large-k distribution. A sum of k
    values of the ARMA part has variance k |H(1)|^2 (long-run variance); per-phase sums
    over k/s seasons have Fourier coefficients with E|P_j|^2 = k |H(w_j)|^2 at the seasonal
    frequencies w_j = 2 pi j / s (an extra 1 / (1 - e^{-i w}) after an ordinary integrator)."""
    k = rng.uniform(0, warm_max + 5 * s)
    level0, season0 = 0.0, np.zeros(s)
    if d == 1 and D == 1:
        level0 = rng.standard_normal() * np.sqrt(k) * abs(transfer(phi, theta, Phi, Theta, s, [0.0])[0])
    if D == 1:
        omegas = 2 * np.pi * np.arange(1, s // 2 + 1) / s
        H = transfer(phi, theta, Phi, Theta, s, omegas)
        if d == 1:
            H = H / (1 - np.exp(-1j * omegas))
        amp = np.sqrt(k) * np.abs(H)                                # sqrt(E|P_j|^2) = sqrt(s m |H|^2), m = k / s
        half = (np.arange(1, s // 2 + 1) * 2 == s)                  # the Nyquist coefficient is real
        coef = np.where(half, amp * rng.standard_normal(len(amp)),
                        amp / np.sqrt(2) * (rng.standard_normal(len(amp)) + 1j * rng.standard_normal(len(amp))))
        season0 = np.fft.irfft(np.r_[0.0, coef], s)
    return level0, season0


def starting_states_batch(gen, phi, theta, Phi, Theta, s, d, D, warm_max):
    """`starting_states` for a batch."""
    n, device = len(d), d.device
    r = lambda *shape: torch.randn(*shape, generator=gen, device=device)        # noqa: E731
    k = (warm_max + 5 * s) * torch.rand(n, generator=gen, device=device)
    H0 = transfer_batch(phi, theta, Phi, Theta, s, [0.0])[:, 0].abs().float()
    level0 = r(n) * k.sqrt() * H0 * ((d == 1) & (D == 1)).float()
    if s == 1:
        return level0, torch.zeros(n, 1, device=device)
    omegas = 2 * np.pi * np.arange(1, s // 2 + 1) / s
    H = transfer_batch(phi, theta, Phi, Theta, s, omegas)
    after_integrator = torch.as_tensor(1 / (1 - np.exp(-1j * omegas)), device=device)
    H = torch.where((d == 1).unsqueeze(1), H * after_integrator, H)
    amp = k.double().sqrt().unsqueeze(1) * H.abs()
    half = torch.as_tensor(np.arange(1, s // 2 + 1) * 2 == s, device=device)
    m = len(omegas)
    re, im = r(n, m).double(), r(n, m).double()
    coef = torch.where(half, torch.complex(amp * re, torch.zeros_like(re)),
                       torch.complex(amp * re / np.sqrt(2), amp * im / np.sqrt(2)))
    coef = torch.cat([torch.zeros(n, 1, dtype=coef.dtype, device=device), coef], 1)
    season0 = torch.fft.irfft(coef, s).float() * (D == 1).float().unsqueeze(1)
    return level0, season0


# ---------------------------------------------------------------------------------------
# GARCH(1,1): a joint prior inside the stationary region
# ---------------------------------------------------------------------------------------
@dataclass(eq=False)
class Garch11(Joint):
    """GARCH(1,1) coefficients drawn through their meaning rather than independently:
    the unconditional variance v, the persistence rho = alpha + beta and the share s of it
    that reacts to news, each uniform on its range; then

        omega = v (1 - rho),   alpha = rho s,   beta = rho (1 - s).

    Every draw has omega > 0, alpha, beta >= 0 and alpha + beta = rho < 1, so (with
    standardized shocks) a finite unconditional variance v; independent draws of the three
    coefficients can leave that region. Use as

        g = tg.priors.Garch11()
        tg.garch(omega=g["omega"], alpha=g["alpha"], beta=g["beta"])
    """
    variance: tuple[float, float] = (0.5, 2.0)
    persistence: tuple[float, float] = (0.6, 0.95)
    share: tuple[float, float] = (0.1, 0.5)

    keys = ("omega", "alpha", "beta", "variance", "persistence", "share")

    def __post_init__(self):
        for name, (lo, hi), (least, most) in (("variance", self.variance, (0, np.inf)),
                                              ("persistence", self.persistence, (0, 0.999)),
                                              ("share", self.share, (0, 1))):
            if not (least <= lo <= hi <= most and np.isfinite(hi)) or (name == "variance" and lo <= 0):
                raise ValueError(f"Garch11 {name} range must be finite, ordered and inside "
                                 f"{'(0, inf)' if name == 'variance' else f'[{least}, {most}]'}")
        # the fast backend samples in float32: every omega must be a positive normal float32,
        # and variances stay below float32's maximum by a factor 1e6 (a conservative margin
        # for the recursion's products, not the exact representability boundary)
        f32 = np.finfo(np.float32)
        smallest_omega = self.variance[0] * (1 - self.persistence[1])
        if smallest_omega < float(f32.tiny) or self.variance[1] > float(f32.max) / 1e6:
            raise ValueError(f"Garch11 variance range {self.variance} is not representable in float32 "
                             f"(smallest omega {smallest_omega:.3g})")

    def describe(self):
        return f"garch11(variance={self.variance}, persistence={self.persistence}, share={self.share})"

    @staticmethod
    def _coefficients(v, rho, s):
        return {"variance": v, "persistence": rho, "share": s,
                "omega": v * (1 - rho), "alpha": rho * s, "beta": rho * (1 - s)}

    def sample(self, rng, draw):
        return self._coefficients(*(rng.uniform(lo, hi) for lo, hi in (self.variance, self.persistence, self.share)))

    def sample_batch(self, gen, n, device, draw):
        return self._coefficients(*(lo + (hi - lo) * torch.rand(n, generator=gen, device=device)
                                    for lo, hi in (self.variance, self.persistence, self.share)))


# ---------------------------------------------------------------------------------------
# earlier names of the ready-made worlds (now in tg.worlds), kept for existing scripts
# ---------------------------------------------------------------------------------------
def low_informative(freq: str, **overrides):
    from .worlds import low_informative as world
    return world(freq, **overrides)


def backbone_mixture():
    from .worlds import backbone
    return backbone()
