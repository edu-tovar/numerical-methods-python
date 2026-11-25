import numpy as np
import pytest

from numerical_methods.floating_point import convertir_binario
from numerical_methods.root_finding import biseccion, newton, secante

SQRT2 = np.sqrt(2)


def f(x):
    return x**2 - 2


def df(x):
    return 2 * x


# ---------------------------------------------------------------- binary expansion

def test_binary_expansion_of_0_1_is_periodic():
    # 0.1 = 0.0001100110011... in base 2: the first 24 bits
    assert convertir_binario(0.1) == "0.000110011001100110011001"
    assert convertir_binario(0.1) == "0." + format(int(0.1 * 2**24), "024b")


def test_binary_expansion_of_0_625_is_exact():
    # 0.625 = 1/2 + 1/8
    assert convertir_binario(0.625) == "0.101"


def test_binary_expansion_with_integer_part():
    assert convertir_binario(5.5) == "101.1"


# ---------------------------------------------------------------- fixed number of iterations (tol=None)

def test_default_three_iterations():
    assert biseccion(f, 0, 2) == 1.375
    assert secante(f, 0, 2) == pytest.approx(10 / 7)
    assert newton(f, df, 2.0) == pytest.approx(577 / 408)  # 2 -> 3/2 -> 17/12 -> 577/408


def test_tol_none_keeps_original_behaviour():
    for n in (1, 5):
        assert biseccion(f, 0, 2, n) == biseccion(f, 0, 2, n, tol=None)
        assert secante(f, 0, 2, n) == secante(f, 0, 2, n, tol=None)
        assert newton(f, df, 2.0, n) == newton(f, df, 2.0, n, tol=None)


def test_secant_without_tolerance_fails_after_convergence():
    # With a fixed number of iterations the secant keeps going after it has
    # converged; f(x_m) == f(x_n) and the method raises. A tolerance stops it first.
    with pytest.raises(ValueError):
        secante(f, 0, 2, n=20)
    assert secante(f, 0, 2, n=20, tol=1e-10) == pytest.approx(SQRT2, abs=1e-12)


# ---------------------------------------------------------------- tolerance

def test_bisection_with_tolerance():
    root = biseccion(f, 0, 2, n=100, tol=1e-10)
    assert abs(root - SQRT2) < 1e-10


def test_bisection_stops_early():
    root = biseccion(f, 0, 2, n=100, tol=1e-3)
    assert 1e-6 < abs(root - SQRT2) < 1e-3


def test_secant_with_tolerance():
    assert secante(f, 0, 2, n=100, tol=1e-10) == pytest.approx(SQRT2, abs=1e-12)


def test_newton_with_tolerance():
    assert newton(f, df, 2.0, n=100, tol=1e-10) == pytest.approx(SQRT2, abs=1e-12)


def test_tolerance_reduces_function_evaluations():
    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        return f(x)

    newton(counted, df, 2.0, n=50)
    without_tol = calls["n"]
    calls["n"] = 0
    newton(counted, df, 2.0, n=50, tol=1e-12)
    assert calls["n"] < without_tol


def test_bisection_requires_sign_change():
    with pytest.raises(ValueError):
        biseccion(f, 2, 3)
