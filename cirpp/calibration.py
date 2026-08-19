"""Calibration CIR++ :

Étape 1 — MLE exact du CIR (kappa, theta, sigma) sur l'historique SOFR overnight,
          via la densité de transition chi-2 non centrale (scipy.stats.ncx2),
          initialisé par une régression AR(1) (pseudo-MLE gaussien).
Étape 2 — Bootstrap des discount factors depuis les par swap rates SOFR OIS
          (single curve, fixed leg annuel ACT/360) puis construction de phi(t).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import ncx2

from .model import CIRParams, CIRPPModel, DiscountCurve, PhiFunction

DAYS_PER_YEAR = 365.0  # axe temps du modèle : ACT/365F


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_sofr(path: str, eps: float = 1e-4) -> pd.DataFrame:
    """Historique SOFR (format FRED : observation_date, SOFR en %).

    Retourne un DataFrame indexé par date avec la colonne ``rate`` en décimal,
    valeurs manquantes supprimées, floor ``eps`` appliqué (le CIR exige x > 0 ;
    la période 2020-2022 contient des fixings quasi nuls).
    """
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", df.columns[1]: "rate"})
    df = df.dropna(subset=["rate"]).set_index("date").sort_index()
    df["rate"] = df["rate"] / 100.0          # % -> décimal
    df["n_floored"] = df["rate"] < eps
    df["rate"] = df["rate"].clip(lower=eps)  # floor epsilon
    return df


def load_swap_rates(path: str) -> tuple[dict[float, float], float | None]:
    """Quotes de marché, une par ligne : ``"ON : 3.62%"``, ``"3M : 4.00%"``,
    ``"2Y : 3.961%"`` (décimales ``.`` ou ``,``).

    Retourne (swaps, spot) : swaps = {ténor en années: taux décimal}, les
    ténors en mois valant m/12 ; spot = fixing SOFR overnight (décimal) si une
    ligne ``ON`` est présente, sinon None.
    """
    swaps: dict[float, float] = {}
    spot: float | None = None
    pattern = re.compile(r"(ON|\d+(?:[.,]\d+)?\s*[MY])\s*:\s*~?\s*([\d.,]+)\s*%")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            rate = float(m.group(2).replace(",", ".")) / 100.0
            tag = m.group(1).replace(" ", "")
            if tag == "ON":
                spot = rate
            elif tag.endswith("M"):
                swaps[float(tag[:-1].replace(",", ".")) / 12.0] = rate
            else:
                swaps[float(tag[:-1].replace(",", "."))] = rate
    if not swaps:
        raise ValueError(f"no swap rates parsed from {path}")
    return dict(sorted(swaps.items())), spot


def tenor_label(tenor_years: float) -> str:
    """0.25 -> '3M', 2.0 -> '2Y'."""
    if tenor_years < 1.0:
        return f"{int(round(tenor_years * 12))}M"
    return f"{tenor_years:g}Y"


# ---------------------------------------------------------------------------
# Étape 1 — MLE exact CIR
# ---------------------------------------------------------------------------

@dataclass
class CIRFitResult:
    params: CIRParams
    loglik: float
    n_obs: int
    init: tuple[float, float, float]  # (kappa, theta, sigma) AR(1)
    converged: bool


def ar1_pseudo_mle(x: np.ndarray, dt: np.ndarray) -> tuple[float, float, float]:
    """Initialisation par régression AR(1) : x_{i+1} = a + b x_i + e.

    kappa = -ln(b)/dt_moyen, theta = a/(1-b),
    sigma^2 estimé par mean(resid^2 / (x_i dt_i)) (Var[e_i] ~ sigma^2 x_i dt_i).
    """
    x_prev, x_next = x[:-1], x[1:]
    dt_bar = float(np.mean(dt))
    b, a = np.polyfit(x_prev, x_next, 1)
    b = float(np.clip(b, 1e-6, 1.0 - 1e-6))
    kappa = -np.log(b) / dt_bar
    theta = float(a) / (1.0 - b)
    resid = x_next - (a + b * x_prev)
    sigma2 = float(np.mean(resid**2 / (x_prev * dt)))
    kappa = float(np.clip(kappa, 1e-3, 50.0))
    theta = float(np.clip(theta, 1e-5, 0.5))
    sigma = float(np.clip(np.sqrt(sigma2), 1e-4, 5.0))
    return kappa, theta, sigma


def cir_exact_loglik(x: np.ndarray, dt: np.ndarray,
                     kappa: float, theta: float, sigma: float) -> float:
    """Log-vraisemblance exacte des transitions CIR via la densité ncx2.

    Sachant x_i, on a 2 c x_{i+1} ~ ncx2(df, nc) avec
        c  = 2 kappa / (sigma^2 (1 - e^{-kappa dt}))
        df = 4 kappa theta / sigma^2
        nc = 2 c x_i e^{-kappa dt}
    """
    x_prev, x_next = x[:-1], x[1:]
    ekt = np.exp(-kappa * dt)
    c = 2.0 * kappa / (sigma**2 * (1.0 - ekt))
    df = 4.0 * kappa * theta / sigma**2
    nc = 2.0 * c * x_prev * ekt
    ll = np.log(2.0 * c) + ncx2.logpdf(2.0 * c * x_next, df, nc)
    return float(np.sum(ll))


def fit_cir_mle(sofr: pd.DataFrame) -> CIRFitResult:
    """MLE exact sur tout l'historique fourni (Nelder-Mead sur les
    log-paramètres), initialisé par AR(1)."""
    x = sofr["rate"].to_numpy()
    days = np.diff(sofr.index.to_numpy().astype("datetime64[D]").astype(float))
    dt = days / DAYS_PER_YEAR

    init = ar1_pseudo_mle(x, dt)

    def neg_ll(log_p: np.ndarray) -> float:
        kappa, theta, sigma = np.exp(log_p)
        ll = cir_exact_loglik(x, dt, kappa, theta, sigma)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg_ll, np.log(init), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    kappa, theta, sigma = np.exp(res.x)
    x0 = float(sofr["rate"].iloc[-1])  # dernier fixing observé
    return CIRFitResult(
        params=CIRParams(kappa=float(kappa), theta=float(theta),
                         sigma=float(sigma), x0=x0),
        loglik=-float(res.fun), n_obs=len(x),
        init=init, converged=bool(res.success),
    )


# ---------------------------------------------------------------------------
# Étape 2 — Bootstrap de la courbe depuis les par swap rates
# ---------------------------------------------------------------------------

def build_swap_schedule(tenor_years: float, valuation_date: pd.Timestamp):
    """Échéancier fixe d'un OIS SOFR : annuel pour les ténors >= 1Y, paiement
    unique à maturité pour les ténors monétaires < 1Y (1M, 3M, 6M...).

    Retourne (times, accruals) : times en années ACT/365F depuis la valuation
    date, accruals fixed leg en ACT/360 entre dates de paiement consécutives.
    Pas d'ajustement business-day (pas de calendrier de jours fériés).
    """
    if tenor_years < 1.0:
        months = int(round(tenor_years * 12))
        dates = [valuation_date, valuation_date + pd.DateOffset(months=months)]
    else:
        n = int(round(tenor_years))
        dates = [valuation_date + pd.DateOffset(years=i) for i in range(n + 1)]
    times = np.array([(d - valuation_date).days / DAYS_PER_YEAR for d in dates[1:]])
    accruals = np.array([(dates[i + 1] - dates[i]).days / 360.0
                         for i in range(len(dates) - 1)])
    return times, accruals


def par_rate_from_dfs(dfs: np.ndarray, accruals: np.ndarray) -> float:
    """Par rate single-curve : S = (1 - P(0, T_N)) / sum_i tau_i P(0, T_i)."""
    annuity = float(np.sum(accruals * dfs))
    return (1.0 - float(dfs[-1])) / annuity


def bootstrap_curve(swaps: dict[float, float], valuation_date: pd.Timestamp,
                    spot_rate: float | None = None) -> DiscountCurve:
    """Bootstrap séquentiel des DF aux piliers, log-linéaire entre piliers.

    Pour chaque ténor, on résout par brentq le DF au pilier tel que le swap
    price au par, les paiements intermédiaires étant interpolés log-
    linéairement entre le pilier précédent et le pilier courant.

    Si ``spot_rate`` (fixing SOFR O/N, décimal) est fourni, un pilier
    overnight est ajouté en t = 1 jour : P = 1 / (1 + spot/360), ce qui ancre
    le forward instantané très court terme sur le fixing du jour.
    """
    pillar_times = [0.0]
    pillar_logdfs = [0.0]
    if spot_rate is not None:
        pillar_times.append(1.0 / DAYS_PER_YEAR)
        pillar_logdfs.append(float(-np.log1p(spot_rate / 360.0)))

    for tenor, rate in swaps.items():
        times, accruals = build_swap_schedule(tenor, valuation_date)
        T = times[-1]

        def pricing_error(df_pillar: float) -> float:
            t = np.array(pillar_times + [T])
            ld = np.array(pillar_logdfs + [np.log(df_pillar)])
            dfs = np.exp(np.interp(times, t, ld))
            return rate * float(np.sum(accruals * dfs)) - (1.0 - dfs[-1])

        df_T = brentq(pricing_error, 1e-8, 1.5, xtol=1e-16, rtol=1e-15)
        pillar_times.append(float(T))
        pillar_logdfs.append(float(np.log(df_T)))

    return DiscountCurve(np.array(pillar_times), np.exp(pillar_logdfs))


# ---------------------------------------------------------------------------
# Vérification du repricing
# ---------------------------------------------------------------------------

def verify_repricing(model: CIRPPModel, swaps: dict[float, float],
                     valuation_date: pd.Timestamp) -> pd.DataFrame:
    """Tableau d'erreurs : ZCB aux piliers (attendu < 1e-10) et par swap
    rates recalculés avec les DF du modèle CIR++ (attendu < 0.1 bp)."""
    rows = []
    curve = model.curve
    for tenor, rate in swaps.items():
        times, accruals = build_swap_schedule(tenor, valuation_date)
        T = times[-1]
        p_mkt = float(curve.df(T))
        p_mod = float(model.zcb_price(T))
        model_dfs = np.asarray(model.zcb_price(times))
        swap_model = par_rate_from_dfs(model_dfs, accruals)
        rows.append({
            "tenor": tenor_label(tenor),
            "T (ACT/365)": T,
            "P_market": p_mkt,
            "P_model": p_mod,
            "zcb_abs_err": abs(p_mod - p_mkt),
            "swap_market_%": rate * 100,
            "swap_model_%": swap_model * 100,
            "swap_err_bp": abs(swap_model - rate) * 1e4,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def build_cirpp_model(params: CIRParams, curve: DiscountCurve,
                      t_max: float = 30.0, smooth_days: float = 10.0) -> CIRPPModel:
    phi = PhiFunction(params, curve, t_max=t_max, smooth_days=smooth_days)
    return CIRPPModel(params, phi)
