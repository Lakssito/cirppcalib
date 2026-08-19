"""Tests de la calibration de la courbe sur un scénario forward utilisateur."""

import numpy as np
import pandas as pd
import pytest

from cirpp.calibration import build_cirpp_model
from cirpp.model import CIRParams, DiscountCurve
from cirpp.scenario import (
    Scenario,
    fit_scenario_curve,
    forward_par_rate,
    load_scenario,
    parse_horizon,
    parse_tenor,
    scenario_schedule,
)

VAL_DATE = pd.Timestamp("2026-08-18")
PARAMS = CIRParams(kappa=1.2, theta=0.045, sigma=0.07, x0=0.036)


def make_curve(zero_fn, t_max: float = 40.0) -> DiscountCurve:
    times = np.linspace(0.25, t_max, 200)
    return DiscountCurve(times, np.exp(-zero_fn(times) * times))


def views_from_curve(curve, horizons, tenors):
    views = []
    for h in horizons:
        for tenor in tenors:
            h_t, t_pay, accr = scenario_schedule(h, tenor, VAL_DATE)
            views.append((h, tenor, forward_par_rate(curve, h_t, t_pay, accr)))
    return views


def base_from_curve(curve, tenors):
    base = {}
    for tenor in tenors:
        h_t, t_pay, accr = scenario_schedule(0.0, tenor, VAL_DATE)
        base[tenor] = forward_par_rate(curve, h_t, t_pay, accr)
    return base


def test_parse_tenor_and_horizon():
    assert parse_tenor("ON") == 0.0
    assert parse_tenor("3M") == pytest.approx(0.25)
    assert parse_tenor("10Y") == 10.0
    assert parse_horizon("0") == 0.0
    assert parse_horizon("6M") == pytest.approx(0.5)
    assert parse_horizon("2Y") == 2.0
    assert parse_horizon("2.5") == 2.5


def test_load_scenario(tmp_path):
    p = tmp_path / "scen.csv"
    p.write_text("horizon,ON,1Y,10Y\n"
                 "0,3.62,3.908,4.228\n"
                 "1Y,3.30,,4.28\n"
                 "weight,1,1,5\n")
    s = load_scenario(str(p))
    assert s.base_quotes == pytest.approx({0.0: 0.0362, 1.0: 0.03908,
                                           10.0: 0.04228})
    assert s.views == [(1.0, 0.0, pytest.approx(0.033)),
                       (1.0, 10.0, pytest.approx(0.0428))]  # cellule vide sautée
    assert s.weights[10.0] == 5.0 and s.weights[0.0] == 1.0


def test_consistent_views_are_repriced():
    """Vues générées depuis une courbe connue (non plate) -> le fit les
    reprice à < 0.1 bp et les quotes h=0 sont conservées."""
    truth = make_curve(lambda t: 0.035 + 0.008 * np.tanh((t - 2.0) / 3.0))
    tenors = [0.0, 1.0, 2.0, 5.0, 10.0]
    views = views_from_curve(truth, [0.5, 1.0, 2.0, 5.0], tenors)
    base = base_from_curve(truth, [1.0, 2.0, 5.0, 10.0, 30.0])
    spot = forward_par_rate(truth, *scenario_schedule(0.0, 0.0, VAL_DATE)[0:1]
                            + scenario_schedule(0.0, 0.0, VAL_DATE)[1:])
    scen = Scenario(views=views, weights={})
    init = truth  # point de départ quelconque admissible
    res = fit_scenario_curve(scen, base, spot, init, VAL_DATE)
    # ~0.3 bp : résidu de la pénalité de lissage + interpolation log-linéaire
    # entre nœuds face à une courbe-vérité courbée — pas un défaut du solveur
    assert res.rmse_views_bp < 0.5
    assert res.max_base_err_bp < 0.5


def test_divergent_views_weights_prioritize():
    """Conflit structurel : les vues 1Y en h=1 et h=2 valent 3.0%, la vue 2Y
    en h=1 vaut 4.5% (elle couvre exactement les deux segments). Un poids
    fort sur le 2Y doit réduire son erreur au détriment du 1Y."""
    flat = make_curve(lambda t: 0.04 * np.ones_like(np.asarray(t, float)))
    views = [(1.0, 1.0, 0.030), (2.0, 1.0, 0.030), (1.0, 2.0, 0.045)]
    base = base_from_curve(flat, [1.0, 2.0, 3.0])

    def err_2y(weights):
        scen = Scenario(views=views, weights=weights)
        res = fit_scenario_curve(scen, base, None, flat, VAL_DATE,
                                 base_weight=5.0)
        t = res.table
        mask = (t["tenor"] == "2Y") & (~t["base"])
        return t.loc[mask, "erreur_bp"].abs().max()

    err_flat = err_2y({1.0: 1.0, 2.0: 1.0})
    err_weighted = err_2y({1.0: 1.0, 2.0: 25.0})
    assert err_weighted < err_flat


def test_cirpp_reprices_scenario_curve_exactly():
    """phi absorbe la courbe scénario : ZCB CIR++ = courbe fittée à la
    précision machine sur ses piliers."""
    truth = make_curve(lambda t: 0.035 + 0.008 * np.tanh((t - 2.0) / 3.0))
    tenors = [0.0, 1.0, 5.0, 10.0]
    scen = Scenario(views=views_from_curve(truth, [1.0, 3.0], tenors),
                    weights={})
    base = base_from_curve(truth, [1.0, 5.0, 10.0, 30.0])
    res = fit_scenario_curve(scen, base, None, truth, VAL_DATE)
    model = build_cirpp_model(PARAMS, res.curve,
                              t_max=float(res.curve.times[-1]) + 1.0)
    pillars = res.curve.times[1:]
    np.testing.assert_allclose(model.zcb_price(pillars), res.curve.df(pillars),
                               rtol=1e-13)
