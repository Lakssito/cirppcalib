"""Calibration CIR++ (Brigo & Mercurio) : r(t) = x(t) + phi(t), x CIR.

Sorties réutilisables dans un framework de simulation externe :
    CIRParams  — kappa, theta, sigma, x0
    PhiFunction — phi(t) évaluable (et Phi(t) = int phi exacte)
    CIRPPModel — bundle sérialisable JSON (CIRPPModel.save / CIRPPModel.load)
"""

from .calibration import (
    bootstrap_curve,
    build_cirpp_model,
    fit_cir_mle,
    load_sofr,
    load_swap_rates,
    verify_repricing,
)
from .model import (
    CIRParams,
    CIRPPModel,
    DiscountCurve,
    PhiFunction,
    cir_A,
    cir_B,
    cir_forward,
    cir_stationary_dist,
    cir_zcb_price,
)

__all__ = [
    "CIRParams", "CIRPPModel", "DiscountCurve", "PhiFunction",
    "cir_A", "cir_B", "cir_forward", "cir_stationary_dist", "cir_zcb_price",
    "load_sofr", "load_swap_rates", "fit_cir_mle", "bootstrap_curve",
    "build_cirpp_model", "verify_repricing",
]
