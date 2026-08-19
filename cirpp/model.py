"""Formules fermées CIR / CIR++ (Brigo & Mercurio, ch. 3.9).

r(t) = x(t) + phi(t), avec dx = kappa (theta - x) dt + sigma sqrt(x) dW.

Le ZCB CIR s'écrit P^CIR(0, T) = A(T) exp(-B(T) x0) avec

    h    = sqrt(kappa^2 + 2 sigma^2)
    B(T) = 2 (e^{hT} - 1) / (2h + (kappa + h)(e^{hT} - 1))
    A(T) = [2h e^{(kappa+h)T/2} / (2h + (kappa + h)(e^{hT} - 1))]^{2 kappa theta / sigma^2}

et le forward instantané CIR (B&M eq. 3.77) :

    f^CIR(0, t) = 2 kappa theta (e^{ht} - 1) / D(t) + x0 4 h^2 e^{ht} / D(t)^2,
    D(t) = 2h + (kappa + h)(e^{ht} - 1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gamma as gamma_dist


# ---------------------------------------------------------------------------
# Paramètres CIR
# ---------------------------------------------------------------------------

@dataclass
class CIRParams:
    kappa: float
    theta: float
    sigma: float
    x0: float

    @property
    def h(self) -> float:
        return float(np.sqrt(self.kappa**2 + 2.0 * self.sigma**2))

    def feller_lhs(self) -> float:
        return 2.0 * self.kappa * self.theta

    def feller_ok(self) -> bool:
        return self.feller_lhs() >= self.sigma**2

    # -- sérialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CIRParams":
        return cls(kappa=d["kappa"], theta=d["theta"], sigma=d["sigma"], x0=d["x0"])


# ---------------------------------------------------------------------------
# Formules fermées CIR
# ---------------------------------------------------------------------------

def cir_B(tau, kappa: float, sigma: float):
    """B(tau) du ZCB CIR."""
    tau = np.asarray(tau, dtype=float)
    h = np.sqrt(kappa**2 + 2.0 * sigma**2)
    e = np.expm1(h * tau)
    return 2.0 * e / (2.0 * h + (kappa + h) * e)


def cir_A(tau, kappa: float, theta: float, sigma: float):
    """A(tau) du ZCB CIR (calculé en log pour la stabilité aux grands tau)."""
    tau = np.asarray(tau, dtype=float)
    h = np.sqrt(kappa**2 + 2.0 * sigma**2)
    e = np.expm1(h * tau)
    log_num = np.log(2.0 * h) + 0.5 * (kappa + h) * tau
    log_den = np.log(2.0 * h + (kappa + h) * e)
    return np.exp(2.0 * kappa * theta / sigma**2 * (log_num - log_den))


def cir_zcb_log_price(tau, params: CIRParams):
    """ln P^CIR(0, tau)."""
    tau = np.asarray(tau, dtype=float)
    return np.log(cir_A(tau, params.kappa, params.theta, params.sigma)) \
        - cir_B(tau, params.kappa, params.sigma) * params.x0


def cir_zcb_price(tau, params: CIRParams):
    """P^CIR(0, tau) = A(tau) exp(-B(tau) x0)."""
    return np.exp(cir_zcb_log_price(tau, params))


def cir_forward(tau, params: CIRParams):
    """Forward instantané f^CIR(0, tau), formule fermée B&M (3.77)."""
    tau = np.asarray(tau, dtype=float)
    kappa, theta, sigma, x0 = params.kappa, params.theta, params.sigma, params.x0
    h = params.h
    e = np.expm1(h * tau)
    den = 2.0 * h + (kappa + h) * e
    return 2.0 * kappa * theta * e / den + x0 * 4.0 * h**2 * np.exp(h * tau) / den**2


def cir_stationary_dist(params: CIRParams):
    """Loi stationnaire du CIR : Gamma(shape=2 kappa theta / sigma^2, scale=sigma^2 / (2 kappa))."""
    shape = 2.0 * params.kappa * params.theta / params.sigma**2
    scale = params.sigma**2 / (2.0 * params.kappa)
    return gamma_dist(a=shape, scale=scale)


# ---------------------------------------------------------------------------
# Courbe de discount marché (interpolation log-linéaire sur les DF)
# ---------------------------------------------------------------------------

class DiscountCurve:
    """Courbe P_market(0, t), log-linéaire entre les piliers.

    ln P est linéaire par morceaux en t (forwards constants par morceaux) ;
    extrapolation à forward instantané constant au-delà du dernier pilier.
    """

    def __init__(self, times, dfs):
        times = np.asarray(times, dtype=float)
        dfs = np.asarray(dfs, dtype=float)
        if times[0] != 0.0:
            times = np.concatenate([[0.0], times])
            dfs = np.concatenate([[1.0], dfs])
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        self.times = times
        self.log_dfs = np.log(dfs)

    def log_df(self, t):
        t = np.asarray(t, dtype=float)
        out = np.interp(t, self.times, self.log_dfs)
        # extrapolation flat-forward au-delà du dernier pilier
        last_slope = (self.log_dfs[-1] - self.log_dfs[-2]) / (self.times[-1] - self.times[-2])
        out = np.where(t > self.times[-1],
                       self.log_dfs[-1] + last_slope * (t - self.times[-1]), out)
        return out

    def df(self, t):
        return np.exp(self.log_df(t))

    def zero_rate(self, t):
        """Taux zéro-coupon continûment composé, -ln P / t."""
        t = np.asarray(t, dtype=float)
        return -self.log_df(t) / np.where(t > 0, t, np.nan)

    def to_dict(self) -> dict:
        return {"times": self.times.tolist(), "dfs": np.exp(self.log_dfs).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DiscountCurve":
        return cls(d["times"], d["dfs"])


# ---------------------------------------------------------------------------
# phi(t) : décalage déterministe du CIR++
# ---------------------------------------------------------------------------

class PhiFunction:
    """phi(t) = f_market(0, t) - f_CIR(0, t).

    Deux représentations cohérentes :

    * ``phi(t)`` / ``phi.value(t)`` — valeur ponctuelle, avec f_market calculé
      numériquement sur une grille fine (pas journalier) puis lissé par noyau
      gaussien : c'est la fonction évaluable destinée à un framework de
      simulation externe.
    * ``phi.integral(t)`` — Phi(t) = int_0^t phi(s) ds calculée EXACTEMENT via
      Phi(t) = ln P^CIR(0, t) - ln P_market(0, t), ce qui garantit le
      repricing des ZCB de marché à la précision machine.
    """

    def __init__(self, params: CIRParams, curve: DiscountCurve,
                 t_max: float = 30.0, grid_step: float = 1.0 / 365.0,
                 smooth_days: float = 10.0):
        self.params = params
        self.curve = curve
        self.t_max = float(t_max)
        self.grid_step = float(grid_step)
        self.smooth_days = float(smooth_days)

        self.grid = np.arange(0.0, self.t_max + grid_step / 2, grid_step)
        # forward instantané marché : dérivée numérique de -ln P_market sur grille fine
        f_mkt = np.gradient(-curve.log_df(self.grid), self.grid)
        # lissage gaussien (sigma en nombre de points de grille)
        sigma_pts = self.smooth_days * (1.0 / 365.0) / self.grid_step
        f_mkt_smooth = gaussian_filter1d(f_mkt, sigma=sigma_pts, mode="nearest")
        self._f_mkt_smooth = f_mkt_smooth
        self._phi_grid = f_mkt_smooth - cir_forward(self.grid, params)

    def value(self, t):
        """phi(t) ponctuel (forward marché lissé - forward CIR). Extrapolation plate."""
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.grid, self._phi_grid)

    __call__ = value

    def integral(self, t):
        """Phi(t) = int_0^t phi(s) ds, exact : ln P^CIR(0,t) - ln P_market(0,t)."""
        return cir_zcb_log_price(t, self.params) - self.curve.log_df(t)

    def market_forward(self, t):
        """f_market(0, t) lissé (pour diagnostic / plots)."""
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.grid, self._f_mkt_smooth)

    def to_dict(self) -> dict:
        return {
            "params": self.params.to_dict(),
            "curve": self.curve.to_dict(),
            "t_max": self.t_max,
            "grid_step": self.grid_step,
            "smooth_days": self.smooth_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhiFunction":
        return cls(CIRParams.from_dict(d["params"]), DiscountCurve.from_dict(d["curve"]),
                   t_max=d["t_max"], grid_step=d["grid_step"], smooth_days=d["smooth_days"])


# ---------------------------------------------------------------------------
# Modèle CIR++ complet (paramètres + phi), sérialisable
# ---------------------------------------------------------------------------

class CIRPPModel:
    """Sortie de calibration : (kappa, theta, sigma, x0) + phi(t) évaluable."""

    def __init__(self, params: CIRParams, phi: PhiFunction):
        self.params = params
        self.phi = phi

    @property
    def curve(self) -> DiscountCurve:
        return self.phi.curve

    def zcb_price(self, T):
        """P^CIR++(0, T) = exp(-Phi(T)) P^CIR(0, T). Reproduit P_market par construction."""
        return np.exp(cir_zcb_log_price(T, self.params) - self.phi.integral(T))

    def zero_rate(self, T):
        T = np.asarray(T, dtype=float)
        return -np.log(self.zcb_price(T)) / np.where(T > 0, T, np.nan)

    def short_rate_0(self) -> float:
        """r(0) = x0 + phi(0)."""
        return float(self.params.x0 + self.phi(0.0))

    # -- sérialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return self.phi.to_dict()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "CIRPPModel":
        phi = PhiFunction.from_dict(d)
        return cls(phi.params, phi)

    @classmethod
    def load(cls, path: str) -> "CIRPPModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))
