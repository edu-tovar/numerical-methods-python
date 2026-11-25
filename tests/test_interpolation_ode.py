import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, KroghInterpolator

from numerical_methods.interpolation import build_interpolator_newton, newton_formula
from numerical_methods.ode import Runge4, euler
from numerical_methods.spline import spline


def runge(x):
    return 1 / (1 + x**2)


# ---------------------------------------------------------------- interpolation

def test_newton_interpolates_the_nodes():
    x = np.linspace(-1, 1, 9)
    p, _ = build_interpolator_newton(x, np.exp(x))
    np.testing.assert_allclose(p(x), np.exp(x), atol=1e-12)


def test_newton_matches_scipy_krogh():
    x = np.linspace(-1, 1, 8)
    y = np.sin(3 * x)
    p, _ = build_interpolator_newton(x, y)
    t = np.linspace(-1, 1, 200)
    np.testing.assert_allclose(p(t), KroghInterpolator(x, y)(t), atol=1e-10)


def test_divided_differences_of_a_cubic():
    # The 3rd divided difference of x^3 is 1 and higher ones vanish.
    x = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    coeffs = newton_formula(x, x**3)
    assert coeffs[3] == pytest.approx(1.0)
    assert coeffs[4] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- spline

@pytest.mark.parametrize("x", [np.linspace(-5, 5, 7), np.array([-5, -3.5, -1, 0, 0.4, 2, 5.0])])
def test_spline_matches_scipy_natural_spline(x):
    t = np.linspace(-5, 5, 1001)
    ours = spline(x, runge(x))(t)
    ref = CubicSpline(x, runge(x), bc_type="natural")(t)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_spline_reproduces_a_straight_line():
    x = np.linspace(0, 3, 6)
    t = np.linspace(0, 3, 50)
    np.testing.assert_allclose(spline(x, 2 * x + 1)(t), 2 * t + 1, atol=1e-12)


# ---------------------------------------------------------------- ODE solvers

def logistic_rhs(t, y):
    return y * (1 - y)


def logistic_exact(t, y0=0.1):
    return 1 / (1 + (1 / y0 - 1) * np.exp(-t))


@pytest.mark.parametrize("method,order", [(euler, 1), (Runge4, 4)])
def test_observed_order_of_convergence(method, order):
    errors = []
    for n in (50, 100, 200):
        x, y = method(logistic_rhs, 0, 5, n, 0.1)
        errors.append(np.max(np.abs(y - logistic_exact(x))))
    observed = np.log2(errors[1] / errors[2])
    assert observed == pytest.approx(order, abs=0.15)


def test_rk4_matches_scipy_solve_ivp():
    x, y = Runge4(logistic_rhs, 0, 5, 200, 0.1)
    ref = solve_ivp(logistic_rhs, (0, 5), [0.1], t_eval=x, rtol=1e-10, atol=1e-12).y[0]
    np.testing.assert_allclose(y, ref, atol=1e-8)
