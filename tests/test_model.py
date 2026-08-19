"""Tests des formules fermées CIR / CIR++."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import ncx2

from cirpp.model import (
    CIRParams,
    cir_A,
    cir_B,
    cir_forward,
    cir_stationary_dist,
    cir_zcb_price,
)

PARAMS = CIRParams(kappa=0.35, theta=0.035, sigma=0.09, x0=0.042)


def test_limits_at_zero():
    assert cir_B(0.0, PARAMS.kappa, PARAMS.sigma) == pytest.approx(0.0, abs=1e-14)
    assert cir_A(0.0, PARAMS.kappa, PARAMS.theta, PARAMS.sigma) == pytest.approx(1.0, abs=1e-14)
    assert cir_zcb_price(0.0, PARAMS) == pytest.approx(1.0, abs=1e-14)
    # f_CIR(0, 0) = x0
    assert cir_forward(0.0, PARAMS) == pytest.approx(PARAMS.x0, abs=1e-14)


def test_zcb_vs_riccati_ode():
    """P(0,T) = A e^{-B x0} doit coïncider avec l'intégration numérique des
    ODE de Riccati : B' = 1 - kappa B - 0.5 sigma^2 B^2, (ln A)' = -kappa theta B."""
    k, th, s = PARAMS.kappa, PARAMS.theta, PARAMS.sigma

    def rhs(_, y):
        B, lnA = y
        return [1.0 - k * B - 0.5 * s**2 * B**2, -k * th * B]

    for T in [0.5, 1.0, 5.0, 10.0, 30.0]:
        sol = solve_ivp(rhs, [0.0, T], [0.0, 0.0], rtol=1e-12, atol=1e-14,
                        dense_output=True)
        B_num, lnA_num = sol.y[:, -1]
        assert cir_B(T, k, s) == pytest.approx(B_num, rel=1e-9)
        assert np.log(cir_A(T, k, th, s)) == pytest.approx(lnA_num, rel=1e-9, abs=1e-12)


def test_forward_is_derivative_of_log_price():
    """f^CIR(0,t) = -d/dt ln P^CIR(0,t) (différences finies centrées)."""
    eps = 1e-6
    for t in [0.1, 1.0, 5.0, 10.0, 25.0]:
        fd = -(np.log(cir_zcb_price(t + eps, PARAMS))
               - np.log(cir_zcb_price(t - eps, PARAMS))) / (2 * eps)
        assert cir_forward(t, PARAMS) == pytest.approx(fd, rel=1e-7)


def test_stationary_moments():
    dist = cir_stationary_dist(PARAMS)
    assert dist.mean() == pytest.approx(PARAMS.theta, rel=1e-12)
    assert dist.var() == pytest.approx(
        PARAMS.sigma**2 * PARAMS.theta / (2 * PARAMS.kappa), rel=1e-12)


def test_mle_recovers_simulated_params():
    """Le MLE exact doit retrouver les paramètres d'une trajectoire simulée
    par échantillonnage exact des transitions ncx2 (simulation côté test
    uniquement — le module lui-même ne simule pas)."""
    from cirpp.calibration import ar1_pseudo_mle, cir_exact_loglik, fit_cir_mle

    rng = np.random.default_rng(42)
    k, th, s = 0.6, 0.03, 0.07
    n, dt = 4000, 1.0 / 252.0
    ekt = np.exp(-k * dt)
    c = 2.0 * k / (s**2 * (1.0 - ekt))
    df = 4.0 * k * th / s**2
    x = np.empty(n)
    x[0] = th
    for i in range(1, n):
        nc = 2.0 * c * x[i - 1] * ekt
        x[i] = ncx2.rvs(df, nc, random_state=rng) / (2.0 * c)

    # maximisation directe de la vraisemblance exacte
    from scipy.optimize import minimize
    dts = np.full(n - 1, dt)
    init = ar1_pseudo_mle(x, dts)

    def neg_ll(lp):
        ll = cir_exact_loglik(x, dts, *np.exp(lp))
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg_ll, np.log(init), method="Nelder-Mead")
    k_hat, th_hat, s_hat = np.exp(res.x)
    # l'optimum doit dominer les vrais paramètres (sinon le MLE est cassé)
    assert -res.fun >= cir_exact_loglik(x, dts, k, th, s) - 1e-6
    # recovery statistique : kappa/theta bruités sur échantillon fini,
    # sigma très bien identifié par la variation quadratique
    assert k_hat == pytest.approx(k, rel=0.5)
    assert th_hat == pytest.approx(th, rel=0.35)
    assert s_hat == pytest.approx(s, rel=0.05)
