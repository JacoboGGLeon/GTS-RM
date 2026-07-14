"""losses.py – Renovated loss‑function catalogue
================================================
This module centralises all **regression loss functions** used by the
*Scientist* layer.  Every function/class is documented with a clear
one‑liner and an extended explanation so that newcomers can grasp the
mathematical intent without leaving the code.

The API remains 100 % backward‑compatible: you can import losses and use
rmse_loss, mae_loss, LearnableHuberLoss, etc. exactly as before.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

# -----------------------------------------------------------------------------
# Simple functional losses
# -----------------------------------------------------------------------------

def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root‑Mean‑Squared Error (RMSE).

    The square root of the mean of squared residuals.  RMSE is in the same
    units as *y* and heavily penalises large errors.
    """
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (MAE).

    Computes the average absolute difference between predictions and targets.
    Less sensitive to outliers than RMSE.
    """
    return F.l1_loss(y_pred, y_true)


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (MSE).

    Useful when you want to penalise *large* errors more than small ones.
    """
    return F.mse_loss(y_pred, y_true)


def smape_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE is scale‑independent and bounded in [0, 200].  A small *eps*
    prevents division by zero when both true and predicted values are 0.
    """
    num = torch.abs(y_pred - y_true)
    den = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + eps
    return (num / den).mean()


def wmape_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted Mean Absolute Percentage Error (WMAPE).

    Proportion of the sum of absolute errors to the sum of actuals.
    Equivalent to MAE / mean(|y_true|).
    """
    num = torch.abs(y_pred - y_true).sum()
    den = torch.abs(y_true).sum() + eps
    return num / den


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Log-Cosh loss: a smooth alternative combining MAE and MSE.

    For residual e, log(cosh(e)) behaves like e²/2 around zero and like
    abs(e) - log(2) for large absolute errors. This keeps the objective
    differentiable while making it less sensitive to outliers than MSE.
    """
    error = y_pred - y_true
    # Adding a tiny constant stabilises the gradient when error ≈ 0.
    return torch.mean(torch.log(torch.cosh(error + 1e-12)))

# -----------------------------------------------------------------------------
# Parametric / learnable loss
# -----------------------------------------------------------------------------

class LearnableHuberLoss(nn.Module):
    """Huber loss with a trainable *delta* threshold.

    The standard Huber loss transitions from L2 to L1 at a fixed *delta*.
    Here, *delta* itself is treated as a **learnable parameter**, allowing
    the model to find the error scale where the switch should occur.

    Parameters
    ----------
    init_delta : float, default = 1.0
        Initial value for *delta*.  Internally we optimise log_delta to
        keep *delta* strictly positive.
    """

    def __init__(self, init_delta: float = 1.0):
        super().__init__()
        # We store log(delta) so that optimisation happens in (−inf, +inf)
        # but delta = exp(log_delta) is always > 0.
        self.log_delta = nn.Parameter(torch.log(torch.tensor(init_delta)))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute the loss.

        The formula is identical to classic Huber, but with *delta*
        replaced by the **learned** value exp(log_delta).
        """
        delta = torch.exp(self.log_delta)
        error = y_true - y_pred
        abs_err = error.abs()

        # Piecewise definition of Huber.
        loss = torch.where(
            abs_err <= delta,
            0.5 * error.pow(2),                # L2 region
            delta * (abs_err - 0.5 * delta),   # L1 region beyond delta
        )
        return loss.mean()

# -----------------------------------------------------------------------------
# Public export list – makes from losses import * safe & ctrl‑clickable
# -----------------------------------------------------------------------------
__all__: list[str | Callable[..., torch.Tensor] | type[nn.Module]] = [
    "rmse_loss",
    "mae_loss",
    "mse_loss",
    "smape_loss",
    "wmape_loss",
    "log_cosh_loss",
    "LearnableHuberLoss",
]