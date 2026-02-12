from __future__ import annotations

from worker.trainer import Value


def test_value_backward_matches_finite_difference() -> None:
    x = Value(2.5)
    y = x * x + 3.0 * x
    y.backward()

    eps = 1e-6
    x_plus = 2.5 + eps
    x_minus = 2.5 - eps
    y_plus = x_plus * x_plus + 3.0 * x_plus
    y_minus = x_minus * x_minus + 3.0 * x_minus
    numerical_grad = (y_plus - y_minus) / (2 * eps)

    assert abs(x.grad - numerical_grad) < 1e-4
