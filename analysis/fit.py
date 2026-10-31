"""Shared sparsity law: ln(perplexity / dense) = a · (N / N*)^(b1 log10 s).

N and N* are parameter counts in billions. Points at or below a model's own
dense perplexity are omitted by the caller.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares


def solve_reference_law(
    sparsities: np.ndarray,
    log_ratios: np.ndarray,
    billions: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return amplitude, N*, b1, and RMSE.

    The estimated coefficients are a, b0, b1 in a · s^(b0 + b1 log10 N), and
    N* = 10^(-b0 / b1). That is the same fit as a · (N / N*)^(b1 log10 s).
    """
    sparsities = np.asarray(sparsities, dtype=float)
    log_ratios = np.asarray(log_ratios, dtype=float)
    billions = np.asarray(billions, dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        amplitude, offset, slope = theta
        return amplitude * np.power(sparsities, offset + slope * np.log10(billions)) - log_ratios

    best = None
    for guess in (
        [12.5, 4.1, 2.7],
        [11.4, 3.9, 2.6],
        [10.0, 4.0, 1.0],
        [18.0, 5.0, -1.0],
        [8.0, 4.5, 0.8],
    ):
        fitted = least_squares(
            residual,
            np.asarray(guess, dtype=float),
            bounds=([1.0e-6, -5.0, -10.0], [1.0e4, 30.0, 10.0]),
            max_nfev=4000,
        )
        err = float(np.sqrt(np.mean(np.square(residual(fitted.x)))))
        if best is None or err < best[0]:
            best = (err, fitted.x.copy())
    rmse, coef = best
    amplitude, offset, slope = (float(coef[0]), float(coef[1]), float(coef[2]))
    n_star = 10.0 ** (-offset / slope)
    return amplitude, n_star, slope, rmse


def exponent_at(n_billions: float, n_star: float, b1: float) -> float:
    return b1 * math.log10(n_billions / n_star)


def solve_scale_free_law(
    sparsities: np.ndarray,
    log_ratios: np.ndarray,
    billions: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return α, β, b, and RMSE for ln(ppl / dense) = α · N^β · s^b.

    Size and sparsity factorize. Changing the unit of N only rescales α.
    """
    sparsities = np.asarray(sparsities, dtype=float)
    log_ratios = np.asarray(log_ratios, dtype=float)
    billions = np.asarray(billions, dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        amplitude, exponent, scale = theta
        return amplitude * np.power(billions, scale) * np.power(sparsities, exponent) - log_ratios

    best = None
    for guess in (
        [18.0, 5.0, -0.4],
        [16.0, 4.7, -0.38],
        [12.0, 4.0, -0.3],
        [15.0, 4.6, -0.35],
    ):
        fitted = least_squares(
            residual,
            np.asarray(guess, dtype=float),
            bounds=([0.0, 0.0, -5.0], [1.0e4, 30.0, 5.0]),
            max_nfev=4000,
        )
        err = float(np.sqrt(np.mean(np.square(residual(fitted.x)))))
        if best is None or err < best[0]:
            best = (err, fitted.x.copy())
    rmse, coef = best
    return float(coef[0]), float(coef[2]), float(coef[1]), rmse


def format_n_star(n_star: float) -> str:
    """Print a reference size. Near-zero or huge values stay in scientific notation."""
    if 0.001 <= n_star < 1000:
        return f"{n_star:.3f}"
    return f"{n_star:.3g}"
