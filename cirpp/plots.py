"""Graphiques de diagnostic (matplotlib, sortie PNG)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .model import CIRPPModel, cir_stationary_dist


def plot_discount_factors(model: CIRPPModel, out_path: str) -> None:
    """P_market(0,T) bootstrappés (points) vs P_model(0,T) CIR++ (ligne)."""
    curve = model.curve
    t_max = curve.times[-1]
    grid = np.linspace(1e-6, t_max, 600)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, model.zcb_price(grid), "-", color="C0", lw=1.5,
            label="P_model(0,T) CIR++")
    ax.plot(curve.times[1:], np.exp(curve.log_dfs[1:]), "o", color="C3", ms=7,
            mfc="none", mew=1.8, label="P_market(0,T) bootstrappés")
    ax.set_xlabel("T (années)")
    ax.set_ylabel("P(0, T)")
    ax.set_title("Discount factors : marché (piliers) vs modèle CIR++")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_zero_rates(model: CIRPPModel, out_path: str) -> None:
    """Taux zéro-coupon marché (piliers) vs modèle (ligne)."""
    curve = model.curve
    t_max = curve.times[-1]
    grid = np.linspace(0.05, t_max, 600)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, 100 * model.zero_rate(grid), "-", color="C0", lw=1.5,
            label="zéro-coupon modèle CIR++")
    t_p = curve.times[1:]
    ax.plot(t_p, 100 * curve.zero_rate(t_p), "o", color="C3", ms=7,
            mfc="none", mew=1.8, label="zéro-coupon marché (piliers)")
    ax.set_xlabel("T (années)")
    ax.set_ylabel("taux zéro (%)")
    ax.set_title("Courbe zéro-coupon : marché vs modèle CIR++")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phi(model: CIRPPModel, out_path: str, t_max: float = 10.0) -> None:
    """phi(t) sur [0, t_max], avec f_market et f_CIR en appui."""
    from .model import cir_forward

    grid = np.linspace(0.0, t_max, 600)
    phi = model.phi
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, 100 * phi(grid), "-", color="C2", lw=2, label="phi(t)")
    ax.plot(grid, 100 * phi.market_forward(grid), "--", color="C0", lw=1,
            label="f_market(0,t) (lissé)")
    ax.plot(grid, 100 * cir_forward(grid, model.params), "--", color="C1", lw=1,
            label="f_CIR(0,t)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t (années)")
    ax.set_ylabel("%")
    ax.set_title("phi(t) = f_market(0,t) - f_CIR(0,t)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scenario_fit(curve, scenario, valuation_date, out_path: str) -> None:
    """Vues forward de l'utilisateur (points) vs taux forward-starting lus
    sur la courbe scénario calibrée (lignes), par ténor, selon l'horizon."""
    from .scenario import forward_par_rate, scenario_schedule, tenor_label

    h_max = max(scenario.horizons) * 1.15
    h_grid = np.linspace(0.0, h_max, 80)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, tenor in enumerate(scenario.tenors):
        color = f"C{i}"
        fitted = []
        for h in h_grid:
            h_t, t_pay, accr = scenario_schedule(h, tenor, valuation_date)
            fitted.append(forward_par_rate(curve, h_t, t_pay, accr))
        w = scenario.weights.get(tenor, 1.0)
        ax.plot(h_grid, 100 * np.array(fitted), "-", color=color, lw=1.5,
                label=f"{tenor_label(tenor)} courbe scénario (poids {w:g})")
        vh = [h for h, t, _ in scenario.views if t == tenor]
        vr = [r for _, t, r in scenario.views if t == tenor]
        ax.plot(vh, 100 * np.array(vr), "o", color=color, ms=8, mfc="none",
                mew=2)
    ax.set_xlabel("horizon h (années)")
    ax.set_ylabel("taux (%)")
    ax.set_title("Vues forward utilisateur (points) vs courbe scénario "
                 "calibrée (lignes)")
    ax.legend(ncols=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_historical_diagnostics(sofr: pd.DataFrame, model: CIRPPModel,
                                out_path: str) -> None:
    """Diagnostic historique : (1) SOFR observé vs densité stationnaire du CIR
    calibré, (2) QQ-plot des incréments standardisés vs N(0,1)."""
    df = sofr
    x = df["rate"].to_numpy()
    p = model.params

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- histogramme vs loi stationnaire
    ax = axes[0]
    dist = cir_stationary_dist(p)
    ax.hist(100 * x, bins=60, density=True, alpha=0.5, color="C0",
            label="SOFR observé")
    grid = np.linspace(1e-6, max(x.max() * 1.3, dist.ppf(0.999)), 400)
    ax.plot(100 * grid, dist.pdf(grid) / 100, "-", color="C3", lw=2,
            label="stationnaire CIR (Gamma)")
    ax.set_xlabel("taux (%)")
    ax.set_ylabel("densité")
    ax.set_title(f"SOFR vs distribution stationnaire "
                 f"({df.index[0].date()} → {df.index[-1].date()})")
    ax.legend()
    ax.grid(alpha=0.3)

    # -- QQ-plot des incréments standardisés
    # (x_{i+1} - x_i - kappa (theta - x_i) dt) / (sigma sqrt(x_i dt)) ~ N(0,1)
    ax = axes[1]
    days = np.diff(df.index.to_numpy().astype("datetime64[D]").astype(float))
    dt = days / 365.0
    x_prev, x_next = x[:-1], x[1:]
    z = (x_next - x_prev - p.kappa * (p.theta - x_prev) * dt) \
        / (p.sigma * np.sqrt(x_prev * dt))
    z = np.sort(z)
    n = len(z)
    q_theo = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(q_theo, z, ".", ms=3, color="C0", label="incréments standardisés")
    lim = [min(q_theo[0], z[0]), max(q_theo[-1], z[-1])]
    ax.plot(lim, lim, "-", color="C3", lw=1, label="y = x")
    ax.set_xlabel("quantiles N(0,1)")
    ax.set_ylabel("quantiles empiriques")
    ax.set_title("QQ-plot des incréments CIR standardisés")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
