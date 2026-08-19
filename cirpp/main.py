"""Autocalibration CIR++ sur l'intégralité des données disponibles.

Enchaîne : MLE exact CIR sur tout l'historique SOFR depuis 2018, bootstrap de
la courbe depuis les quotes du jour (spot O/N + OIS monétaires + OIS annuels),
fit exact de phi(t), vérification du repricing, graphiques, sérialisation.

Usage :
    python -m cirpp.main [--data-dir .] [--eps 1e-4] [--out-dir output]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .calibration import (
    bootstrap_curve,
    build_cirpp_model,
    fit_cir_mle,
    load_sofr,
    load_swap_rates,
    tenor_label,
    verify_repricing,
)
from .plots import (
    plot_discount_factors,
    plot_historical_diagnostics,
    plot_phi,
    plot_zero_rates,
)

SEP = "=" * 74


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Autocalibration CIR++ sur tout l'historique SOFR")
    ap.add_argument("--data-dir", default=".", help="dossier des données")
    ap.add_argument("--sofr-file", default="SOFR.csv")
    ap.add_argument("--swaps-file", default="swapsofrrates.txt")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="floor sur les taux quasi nuls (décimal)")
    ap.add_argument("--valuation-date", default=None,
                    help="date de valorisation de la courbe (défaut : aujourd'hui)")
    ap.add_argument("--smooth-days", type=float, default=10.0,
                    help="écart-type (en jours) du lissage gaussien du forward "
                         "marché dans phi(t) ponctuel ; ~0 = forward brut "
                         "constant par morceaux")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    val_date = (pd.Timestamp(args.valuation_date) if args.valuation_date
                else pd.Timestamp.today().normalize())

    # ---- données ----------------------------------------------------------
    sofr = load_sofr(os.path.join(args.data_dir, args.sofr_file), eps=args.eps)
    swaps, spot = load_swap_rates(os.path.join(args.data_dir, args.swaps_file))
    n_floored = int(sofr["n_floored"].sum())
    print(SEP)
    print("DONNEES")
    print(SEP)
    print(f"SOFR : {len(sofr)} fixings du {sofr.index[0].date()} au "
          f"{sofr.index[-1].date()} ({n_floored} fixings floorés à eps={args.eps})")
    print(f"Quotes du jour ({val_date.date()}) : "
          + ", ".join(f"{tenor_label(t)}={r * 100:.3f}%" for t, r in swaps.items()))
    if spot is not None:
        print(f"Fixing SOFR spot (O/N) : {spot * 100:.3f}% "
              "-> utilisé comme x0 et comme ancre court terme de la courbe")

    # ---- étape 1 : MLE CIR sur tout l'historique --------------------------
    print("\n" + SEP)
    print("ETAPE 1 — MLE exact CIR (densité ncx2) sur tout l'historique")
    print(SEP)
    fit = fit_cir_mle(sofr)
    if spot is not None:
        fit.params.x0 = spot  # x0 = fixing spot du jour
    p = fit.params
    print(f"n_obs             : {fit.n_obs}")
    print(f"kappa             : {p.kappa:.6f}")
    print(f"theta             : {p.theta:.6f}")
    print(f"sigma             : {p.sigma:.6f}")
    print(f"x0                : {p.x0:.6f}"
          + ("  (fixing spot O/N)" if spot is not None else "  (dernier fixing)"))
    print(f"log-vraisemblance : {fit.loglik:.2f}")
    print("init AR(1)        : kappa={:.4f}, theta={:.5f}, sigma={:.5f}"
          .format(*fit.init))
    print(f"convergence MLE   : {fit.converged}")
    feller_ok = p.feller_ok()
    print(f"condition Feller  : 2*kappa*theta = {p.feller_lhs():.4e} "
          f"{'>=' if feller_ok else '<'} sigma^2 = {p.sigma**2:.4e} "
          f"[{'OK' if feller_ok else 'VIOLEE'}]")
    if not feller_ok:
        print("  *** ATTENTION : condition de Feller VIOLEE — le processus "
              "CIR peut toucher 0. ***")

    # ---- étape 2 : bootstrap + phi ---------------------------------------
    print("\n" + SEP)
    print("ETAPE 2 — Bootstrap de la courbe et fit exact de phi(t)")
    print(SEP)
    curve = bootstrap_curve(swaps, val_date, spot_rate=spot)
    labels = (["O/N"] if spot is not None else []) + [tenor_label(t) for t in swaps]
    df_table = pd.DataFrame({
        "T (ACT/365)": curve.times[1:],
        "P_market(0,T)": np.exp(curve.log_dfs[1:]),
        "zero rate (%)": 100 * curve.zero_rate(curve.times[1:]),
    }, index=labels)
    print(df_table.to_string(float_format=lambda v: f"{v:.10f}"))

    model = build_cirpp_model(fit.params, curve, t_max=max(swaps) + 1.0,
                              smooth_days=args.smooth_days)
    print(f"\nphi(0) = {float(model.phi(0.0)) * 1e4:.1f} bp ; "
          f"r(0) = x0 + phi(0) = {model.short_rate_0() * 100:.4f}% "
          f"(lissage {args.smooth_days:g}j du forward marché)")

    # ---- vérification du repricing ---------------------------------------
    print("\n" + SEP)
    print("VERIFICATION DU REPRICING (CIR++ avec phi exact)")
    print(SEP)
    errs = verify_repricing(model, swaps, val_date)
    with pd.option_context("display.float_format", lambda v: f"{v:.12g}"):
        print(errs.to_string(index=False))
    zcb_ok = (errs["zcb_abs_err"] < 1e-10).all()
    swap_ok = (errs["swap_err_bp"] < 0.1).all()
    print(f"\nZCB aux piliers  : max err = {errs['zcb_abs_err'].max():.3e}  "
          f"(seuil 1e-10) -> {'OK' if zcb_ok else 'ECHEC'}")
    print(f"Par swap rates   : max err = {errs['swap_err_bp'].max():.3e} bp "
          f"(seuil 0.1 bp) -> {'OK' if swap_ok else 'ECHEC'}")
    if not (zcb_ok and swap_ok):
        raise RuntimeError("Echec du repricing — calibration invalide.")

    # ---- sérialisation ----------------------------------------------------
    out_json = os.path.join(args.out_dir, "cirpp_model.json")
    model.save(out_json)
    print(f"\nModèle sérialisé : {out_json}")
    print("  Rechargement : from cirpp import CIRPPModel ; "
          "m = CIRPPModel.load(path) ; m.params, m.phi(t)")

    # ---- graphiques -------------------------------------------------------
    plots = {
        "discount_factors.png": lambda path: plot_discount_factors(model, path),
        "zero_rates.png": lambda path: plot_zero_rates(model, path),
        "phi.png": lambda path: plot_phi(model, path, t_max=10.0),
        "historical_diagnostics.png":
            lambda path: plot_historical_diagnostics(sofr, model, path),
    }
    for name, fn in plots.items():
        path = os.path.join(args.out_dir, name)
        fn(path)
        print(f"Graphique : {path}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
