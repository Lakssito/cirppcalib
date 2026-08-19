"""Tests du bootstrap de courbe, du repricing CIR++ et de la sérialisation."""

import numpy as np
import pandas as pd
import pytest

from cirpp.calibration import (
    bootstrap_curve,
    build_cirpp_model,
    build_swap_schedule,
    par_rate_from_dfs,
    verify_repricing,
)
from cirpp.calibration import load_swap_rates
from cirpp.model import CIRParams, CIRPPModel, DiscountCurve


def test_load_swap_rates_parses_months_and_spot(tmp_path):
    p = tmp_path / "quotes.txt"
    p.write_text("ON : 3.62%\n1M : ~3,8%\n6M : 3.83%\n2Y : 3.961%\n")
    swaps, spot = load_swap_rates(str(p))
    assert spot == pytest.approx(0.0362)
    assert swaps == pytest.approx({1.0 / 12: 0.038, 0.5: 0.0383, 2.0: 0.03961})

VAL_DATE = pd.Timestamp("2026-08-18")
TENORS = [1.0 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0]
PARAMS = CIRParams(kappa=0.4, theta=0.035, sigma=0.08, x0=0.036)


def make_par_swaps_from_flat_curve(rate: float) -> tuple[dict, DiscountCurve]:
    """Par rates générés depuis une courbe à forward instantané constant."""
    times = np.array([build_swap_schedule(t, VAL_DATE)[0][-1] for t in TENORS])
    curve = DiscountCurve(times, np.exp(-rate * times))
    swaps = {}
    for tenor in TENORS:
        t_pay, accr = build_swap_schedule(tenor, VAL_DATE)
        swaps[tenor] = par_rate_from_dfs(curve.df(t_pay), accr)
    return swaps, curve


def test_bootstrap_roundtrip():
    """DF connus -> par rates -> bootstrap -> mêmes DF (aux piliers)."""
    swaps, curve_true = make_par_swaps_from_flat_curve(0.04)
    curve_bs = bootstrap_curve(swaps, VAL_DATE)
    np.testing.assert_allclose(np.exp(curve_bs.log_dfs), np.exp(curve_true.log_dfs),
                               rtol=1e-12)


def test_short_tenor_schedule_is_single_payment():
    times, accruals = build_swap_schedule(0.25, VAL_DATE)
    assert len(times) == 1 and len(accruals) == 1
    assert times[0] == pytest.approx(92 / 365.0)   # 2026-08-18 -> 2026-11-18
    assert accruals[0] == pytest.approx(92 / 360.0)


def test_spot_anchor():
    """Le pilier O/N vaut P = 1/(1 + spot/360) en t = 1 jour et le forward
    court terme de la courbe s'ancre sur le fixing spot."""
    swaps, _ = make_par_swaps_from_flat_curve(0.04)
    spot = 0.0362
    curve = bootstrap_curve(swaps, VAL_DATE, spot_rate=spot)
    t_on = 1.0 / 365.0
    assert float(curve.df(t_on)) == pytest.approx(1.0 / (1.0 + spot / 360.0),
                                                  rel=1e-14)
    # forward instantané sur [0, 1j] = spot * 365/360 (simple ACT/360 -> continu)
    f_short = -float(curve.log_df(t_on)) / t_on
    assert f_short == pytest.approx(spot * 365.0 / 360.0, rel=1e-3)


def test_repricing_thresholds():
    """ZCB < 1e-10 aux piliers, swaps < 0.1 bp — les seuils exigés."""
    swaps, _ = make_par_swaps_from_flat_curve(0.04)
    curve = bootstrap_curve(swaps, VAL_DATE)
    model = build_cirpp_model(PARAMS, curve, t_max=31.0)
    errs = verify_repricing(model, swaps, VAL_DATE)
    assert (errs["zcb_abs_err"] < 1e-10).all()
    assert (errs["swap_err_bp"] < 0.1).all()


def test_phi_integral_consistency():
    """Phi(t) exacte vs intégration numérique (trapèzes) du phi ponctuel non
    lissé : cohérence à l'intérieur d'un segment de la courbe (loin des sauts
    de forward, le lissage est inactif)."""
    swaps, _ = make_par_swaps_from_flat_curve(0.04)
    curve = bootstrap_curve(swaps, VAL_DATE)
    model = build_cirpp_model(PARAMS, curve, t_max=31.0, smooth_days=0.001)
    ts = np.linspace(0.0, 10.0, 4001)
    num = np.trapezoid(model.phi(ts), ts)
    exact = float(model.phi.integral(10.0))
    assert num == pytest.approx(exact, abs=5e-5)


def test_model_serialization_roundtrip(tmp_path):
    swaps, _ = make_par_swaps_from_flat_curve(0.04)
    curve = bootstrap_curve(swaps, VAL_DATE)
    model = build_cirpp_model(PARAMS, curve, t_max=31.0)
    path = tmp_path / "model.json"
    model.save(str(path))
    reloaded = CIRPPModel.load(str(path))

    assert reloaded.params.to_dict() == model.params.to_dict()
    ts = np.linspace(0.0, 30.0, 200)
    np.testing.assert_allclose(reloaded.phi(ts), model.phi(ts), rtol=0, atol=1e-15)
    np.testing.assert_allclose(reloaded.zcb_price(ts), model.zcb_price(ts),
                               rtol=1e-14)


def test_zcb_matches_market_at_pillars_machine_precision():
    swaps, _ = make_par_swaps_from_flat_curve(0.04)
    curve = bootstrap_curve(swaps, VAL_DATE)
    model = build_cirpp_model(PARAMS, curve, t_max=31.0)
    pillars = curve.times[1:]
    np.testing.assert_allclose(model.zcb_price(pillars), curve.df(pillars),
                               rtol=1e-13)
