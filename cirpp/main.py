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
    plot_scenario_fit,
    plot_zero_rates,
)
from .scenario import fit_scenario_curve, load_scenario

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
    ap.add_argument("--scenario", default=None,
                    help="fichier CSV de scénario forward utilisateur "
                         "(cf. scenario_example.csv) : la courbe (donc phi) "
                         "est recalibrée sur les vues forward, pondérées par "
                         "ténor")
    ap.add_argument("--base-weight", type=float, default=50.0,
                    help="poids des quotes h=0 dans le fit scénario")
    ap.add_argument("--smooth-lambda", type=float, default=0.05,
                    help="pénalité de lissage des forwards dans le fit "
                         "scénario")
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

    # ---- scénario utilisateur (optionnel) --------------------------------
    scenario = None
    if args.scenario:
        scenario = load_scenario(args.scenario)
        if scenario.base_quotes:
            # la ligne h=0 du scénario remplace les quotes de marché
            base = dict(scenario.base_quotes)
            scen_spot = base.pop(0.0, None)
            if scen_spot is not None:
                spot = scen_spot
                fit.params.x0 = spot
            swaps = base
            print(f"\nScénario : ligne h=0 trouvée -> courbe initiale prise "
                  f"dans {args.scenario}")
        else:
            print(f"\nScénario : pas de ligne h=0 -> courbe initiale prise "
                  f"dans {args.swaps_file}")

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

    # ---- étape 2bis : fit de la courbe sur le scénario -------------------
    if scenario is not None:
        print("\n" + SEP)
        print("ETAPE 2bis — Fit de la courbe (donc de phi) sur le scénario "
              "forward utilisateur")
        print(SEP)
        scen_fit = fit_scenario_curve(scenario, swaps, spot, curve, val_date,
                                      base_weight=args.base_weight,
                                      smooth_lambda=args.smooth_lambda)
        curve = scen_fit.curve
        print("Les vues sont traitées comme instruments forward-starting sur "
              "la courbe ;\nkappa/theta/sigma restent ceux de la calibration "
              "historique, x0 = spot.\n")
        print(scen_fit.table.to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nRMSE pondérée sur les vues   : {scen_fit.rmse_views_bp:.2f} bp")
        print(f"Écart max sur les quotes h=0 : {scen_fit.max_base_err_bp:.4f} bp "
              f"(poids base {args.base_weight:g})")
        print(f"convergence : {scen_fit.converged}")
        if scen_fit.rmse_views_bp > 1.0:
            print("NB : RMSE > 1 bp -> vues mutuellement divergentes, le fit "
                  "est le compromis pondéré.")

    model = build_cirpp_model(fit.params, curve,
                              t_max=float(curve.times[-1]) + 1.0,
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
    if scenario is not None and not swap_ok:
        print("NB : en mode scénario, l'écart aux quotes h=0 est le compromis "
              "pondéré vues/quotes,\npas un échec de calibration (le modèle "
              "reprice EXACTEMENT la courbe scénario).")
        swap_ok = True
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
    if scenario is not None:
        plots["scenario_fit.png"] = lambda path: plot_scenario_fit(
            curve, scenario, val_date, path)
    for name, fn in plots.items():
        path = os.path.join(args.out_dir, name)
        fn(path)
        print(f"Graphique : {path}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
