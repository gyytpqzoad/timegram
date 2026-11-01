"""Readable names for new compositions; no new simulation engine.

Ported from the grammar_v2_20261028 prototype; the default MA prior is invertible
(`Invertible`, timegram's sign convention).
"""
from .blocks import ar, ma
from .compounds import Compound
from .dists import Invertible, Stationary, as_dist
from .fractional import fractional_integrate


class ARFIMA(Compound):
    name = "arfima"

    def __init__(self, p=0, d=0.3, q=0, phi=None, theta=None, presample=1024, memory=None):
        if isinstance(p, bool) or not isinstance(p, int) or p < 0:
            raise ValueError("p must be a nonnegative integer")
        if isinstance(q, bool) or not isinstance(q, int) or q < 0:
            raise ValueError("q must be a nonnegative integer")
        self.p, self.q, self.d = p, q, as_dist(d)
        self.phi = as_dist(Stationary(p) if phi is None else phi)
        self.theta = as_dist(Invertible(q) if theta is None else theta)
        self.history, self.memory = presample, memory
        super().__init__()

    def build(self, e):
        x = e
        if self.q:
            x = x >> ma(self.q, theta=self.theta)
        if self.p:
            x = x >> ar(self.p, phi=self.phi)
        return x >> fractional_integrate(self.d, presample=self.history, memory=self.memory)

    def params(self):
        parameters = {"d": self.d}
        if self.p:
            parameters["phi"] = self.phi
        if self.q:
            parameters["theta"] = self.theta
        return parameters

    def label(self):
        return f"arfima({self.p}, {self.d.describe()}, {self.q}; prehistory={self.history})"


def arfima(p=0, d=0.3, q=0, **kwargs):
    """ARMA followed by fractional integration; finite prehistory is explicit.

    The default 1024 steps reduces startup effects, but does not certify exact
    stationary initialization. Override presample=0 for a zero-start process.
    """
    return ARFIMA(p, d, q, **kwargs)
