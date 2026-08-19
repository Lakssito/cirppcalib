"""Calibration du CIR++ sur un scénario forward fourni par l'utilisateur.

L'utilisateur fournit, pour chaque ténor (spot O/N, swap 1Y, 2Y, ..., 30Y) et
chaque horizon h (6M, 1Y, 2Y, ...), sa vision du niveau du taux forward. Ces
vues sont traitées comme des instruments FORWARD-STARTING sur la courbe :

    S(h, h+N) = (P(0,h) - P(0,T_N)) / sum_i tau_i P(0,t_i)      (swap NY vu en h)
    ON(h)     = (P(0,h)/P(0,h+1j) - 1) * 360                    (fixing O/N vu en h)

La calibration construit la COURBE scénario par moindres carrés pondérés :

    min_{DF piliers}  sum_vues  w_tenor · (S_courbe - S_vue)^2·(1e4)^2
                    + base_weight · (erreurs sur les quotes h=0)^2
                    + smooth_lambda · (sauts de forwards adjacents)^2

Les vues mutuellement divergentes sont arbitrées par les poids par ténor
(ligne ``weight`` du CSV) : un poids 5 sur le 10Y tire le compromis vers les
forwards du 10Y. Des vues cohérentes entre elles sont repricées quasi
exactement. phi(t) absorbe ensuite EXACTEMENT la courbe scénario (identité
fermée du CIR++), les paramètres de dynamique (kappa, theta, sigma) restant
ceux de la calibration historique et x0 le fixing spot.

Pourquoi ne pas recalibrer (kappa, theta) sur les vues le long du chemin
espéré E[x(h)] ? Parce que par Jensen le taux futur impliqué sur ce chemin est
toujours >= au forward initial absorbé par phi : des vues sous les forwards de
marché sont alors inatteignables et l'optimisation dégénère (kappa explose,
theta -> 0, Feller violée). La bonne inconnue face à des vues de forwards est
la courbe — donc phi.

Format du fichier scénario (CSV, taux en %) :

    horizon,ON,1Y,2Y,5Y,10Y,30Y
    0,3.62,3.908,3.961,4.025,4.228,4.483
    6M,3.45,3.75,3.85,4.00,4.25,4.50
    1Y,3.30,3.60,3.75,3.95,4.28,4.52
    weight,1,1,1,1,5,1

- La ligne ``0`` (optionnelle) définit les quotes initiales du scénario ; à
  défaut, les quotes de swapsofrrates.txt sont utilisées.
- Cellules vides autorisées (vue non contrainte). Plusieurs lignes peuvent
  porter le même horizon (scénarios divergents superposés).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import DiscountCurve

DAYS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Parsing du scénario
# ---------------------------------------------------------------------------

def parse_tenor(label: str) -> float:
    """'ON' -> 0.0, '3M' -> 0.25, '2Y' -> 2.0."""
    s = str(label).strip().upper().replace(" ", "").replace(",", ".")
    if s in ("ON", "O/N"):
        return 0.0
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    raise ValueError(f"ténor illisible : {label!r}")


def parse_horizon(label: str) -> float:
    """'0' -> 0.0, '6M' -> 0.5, '1Y' -> 1.0, '2.5' -> 2.5 (années)."""
    s = str(label).strip().upper().replace(" ", "").replace(",", ".")
    if s in ("0", "0.0", "SPOT", "0Y", "0M"):
        return 0.0
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)


def tenor_label(tenor_years: float) -> str:
    if tenor_years == 0.0:
        return "ON"
    if tenor_years < 1.0:
        return f"{int(round(tenor_years * 12))}M"
    return f"{tenor_years:g}Y"


@dataclass
class Scenario:
    """Vues forward de l'utilisateur.

    views : liste de (horizon_années, ténor_années, taux_décimal), h > 0 ;
            ténor 0.0 = fixing O/N. Doublons autorisés (vues divergentes).
    weights : poids par ténor (défaut 1).
    base_quotes : ligne h=0 (ténor -> taux décimal), quotes initiales du
                  scénario ; clé 0.0 = fixing spot O/N.
    """
    views: list[tuple[float, float, float]]
    weights: dict[float, float]
    base_quotes: dict[float, float] = field(default_factory=dict)

    @property
    def tenors(self) -> list[float]:
        return sorted({t for _, t, _ in self.views})

    @property
    def horizons(self) -> list[float]:
        return sorted({h for h, _, _ in self.views})


def load_scenario(path: str) -> Scenario:
    df = pd.read_csv(path, dtype=str)
    tenor_cols = {col: parse_tenor(col) for col in df.columns[1:]}
    weights = {t: 1.0 for t in tenor_cols.values()}
    views: list[tuple[float, float, float]] = []
    base: dict[float, float] = {}

    def cell(v) -> float | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip().replace("%", "").replace(",", ".").replace("~", "")
        return float(s) if s else None

    for _, row in df.iterrows():
        head = str(row.iloc[0]).strip().lower()
        if head in ("weight", "weights", "poids"):
            for col, t in tenor_cols.items():
                w = cell(row[col])
                if w is not None:
                    weights[t] = w
            continue
        h = parse_horizon(row.iloc[0])
        for col, t in tenor_cols.items():
            v = cell(row[col])
            if v is None:
                continue
            rate = v / 100.0
            if h == 0.0:
                base[t] = rate
            else:
                views.append((h, t, rate))

    if not views:
        raise ValueError(f"aucune vue d'horizon > 0 dans {path}")
    return Scenario(views=views, weights=weights, base_quotes=base)


# ---------------------------------------------------------------------------
# Instruments forward-starting sur la courbe
# ---------------------------------------------------------------------------

def scenario_schedule(horizon_years: float, tenor_years: float,
                      valuation_date: pd.Timestamp):
    """Échéancier du taux vu à l'horizon h : (h_t, t_pay, accruals), temps en
    années ACT/365 depuis la valuation date, accruals ACT/360.

    ténor 0 (ON)  : taux simple sur 1 jour [h, h+1j].
    ténor < 1Y    : OIS monétaire, paiement unique à h + m mois.
    ténor >= 1Y   : OIS annuel, paiements h + 1an, ..., h + N ans.
    """
    d_h = valuation_date + pd.DateOffset(months=int(round(horizon_years * 12)))
    if tenor_years == 0.0:
        dates = [d_h, d_h + pd.Timedelta(days=1)]
    elif tenor_years < 1.0:
        months = int(round(tenor_years * 12))
        dates = [d_h, d_h + pd.DateOffset(months=months)]
    else:
        n = int(round(tenor_years))
        dates = [d_h + pd.DateOffset(years=i) for i in range(n + 1)]
    h_t = (d_h - valuation_date).days / DAYS_PER_YEAR
    t_pay = np.array([(d - valuation_date).days / DAYS_PER_YEAR
                      for d in dates[1:]])
    accruals = np.array([(dates[i + 1] - dates[i]).days / 360.0
                         for i in range(len(dates) - 1)])
    return h_t, t_pay, accruals


def forward_par_rate(curve: DiscountCurve, h_t: float, t_pay: np.ndarray,
                     accruals: np.ndarray) -> float:
    """Par rate forward-starting lu sur la courbe :
    S = (P(0,h) - P(0,T_N)) / sum_i tau_i P(0,t_i). En h=0, par rate spot ;
    pour l'ON, taux simple 1 jour ACT/360."""
    dfs = np.exp(curve.log_df(t_pay))
    df_h = float(np.exp(curve.log_df(h_t)))
    return (df_h - float(dfs[-1])) / float(np.sum(accruals * dfs))


# ---------------------------------------------------------------------------
# Fit de la courbe scénario
# ---------------------------------------------------------------------------

@dataclass
class ScenarioFitResult:
    curve: DiscountCurve      # courbe scénario (à donner à PhiFunction)
    table: pd.DataFrame       # instrument par instrument : cible vs fit
    rmse_views_bp: float      # RMSE pondérée sur les vues h > 0
    max_base_err_bp: float    # écart max sur les quotes h=0
    converged: bool


def fit_scenario_curve(scenario: Scenario, base_quotes: dict[float, float],
                       spot: float | None, init_curve: DiscountCurve,
                       valuation_date: pd.Timestamp,
                       base_weight: float = 50.0,
                       smooth_lambda: float = 0.05) -> ScenarioFitResult:
    """Moindres carrés pondérés sur les log-DF aux piliers du scénario.

    Inconnues : log-DF aux nœuds (horizons et maturités finales de tous les
    instruments, fusionnés au jour près), interpolation log-linéaire entre
    nœuds. Résidus (en bp) : vues (poids w_tenor), quotes h=0 (poids
    ``base_weight``, quasi-contraintes sauf conflit fort avec les vues),
    pénalité de lissage ``smooth_lambda`` sur les sauts de forwards entre
    segments adjacents (régularise les directions non contraintes).
    Point de départ : la courbe bootstrappée des quotes h=0 (``init_curve``).
    """
    # instruments : (h_t, t_pay, accruals, cible, poids, horizon_lbl, tenor_lbl, base?)
    instruments = []
    if spot is not None:
        h_t, t_pay, accr = scenario_schedule(0.0, 0.0, valuation_date)
        instruments.append((h_t, t_pay, accr, spot, base_weight, "0", "ON", True))
    for tenor, rate in sorted(base_quotes.items()):
        h_t, t_pay, accr = scenario_schedule(0.0, tenor, valuation_date)
        instruments.append((h_t, t_pay, accr, rate, base_weight, "0",
                            tenor_label(tenor), True))
    for h, tenor, rate in scenario.views:
        h_t, t_pay, accr = scenario_schedule(h, tenor, valuation_date)
        w = scenario.weights.get(tenor, 1.0)
        instruments.append((h_t, t_pay, accr, rate, w, f"{h:g}Y",
                            tenor_label(tenor), False))

    # nœuds : horizons + maturités finales, fusionnés au jour près
    knot_days = set()
    for h_t, t_pay, _, _, _, _, _, _ in instruments:
        if h_t > 0:
            knot_days.add(int(round(h_t * DAYS_PER_YEAR)))
        knot_days.add(int(round(t_pay[-1] * DAYS_PER_YEAR)))
    knots = np.array(sorted(knot_days)) / DAYS_PER_YEAR

    x_init = init_curve.log_df(knots)
    knots0 = np.concatenate([[0.0], knots])

    # pénalité de lissage type Sobolev : (saut de forward)^2 / écart de temps
    # entre milieux de segments — un spike sur un segment d'1 jour coûte
    # ~sqrt(365) fois plus qu'une inflexion sur un segment annuel, ce qui
    # empêche de fitter les vues O/N par des creux d'un jour aux nœuds.
    seg_mids = 0.5 * (knots0[:-1] + knots0[1:])
    smooth_scale = np.sqrt(smooth_lambda / np.diff(seg_mids))

    def residuals(log_dfs: np.ndarray) -> np.ndarray:
        curve = DiscountCurve(knots, np.exp(log_dfs))
        res = [np.sqrt(w) * (forward_par_rate(curve, h_t, t_pay, accr)
                             - target) * 1e4
               for h_t, t_pay, accr, target, w, _, _, _ in instruments]
        ld0 = np.concatenate([[0.0], log_dfs])
        fwds = -np.diff(ld0) / np.diff(knots0)
        res_smooth = smooth_scale * np.diff(fwds) * 1e4
        return np.concatenate([np.array(res), res_smooth])

    sol = least_squares(residuals, x_init, bounds=(-20.0, 0.5),
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    curve = DiscountCurve(knots, np.exp(sol.x))

    rows, wsum, werr2, base_err = [], 0.0, 0.0, 0.0
    for h_t, t_pay, accr, target, w, h_lbl, t_lbl, is_base in instruments:
        fitted = forward_par_rate(curve, h_t, t_pay, accr)
        err_bp = (fitted - target) * 1e4
        if is_base:
            base_err = max(base_err, abs(err_bp))
        else:
            wsum += w
            werr2 += w * err_bp**2
        rows.append({"horizon": h_lbl, "tenor": t_lbl,
                     "base": is_base, "poids": w,
                     "cible_%": target * 100, "fit_%": fitted * 100,
                     "erreur_bp": err_bp})
    return ScenarioFitResult(
        curve=curve, table=pd.DataFrame(rows),
        rmse_views_bp=float(np.sqrt(werr2 / wsum)) if wsum else 0.0,
        max_base_err_bp=float(base_err), converged=bool(sol.success))
