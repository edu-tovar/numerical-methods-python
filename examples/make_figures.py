"""Reproduce the figures and numbers in the README.

Usage:
    python examples/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

from numerical_methods.interpolation import build_interpolator_newton  # noqa: E402
from numerical_methods.ode import Runge4, euler  # noqa: E402
from numerical_methods.root_finding import biseccion, newton, secante  # noqa: E402
from numerical_methods.spline import spline  # noqa: E402

# ---------------------------------------------------------------- 1. Runge's phenomenon
def runge(x):
    return 1.0 / (1.0 + x**2)


nodes = np.linspace(-10, 10, 30)
t = np.linspace(-10, 10, 4001)
p, _ = build_interpolator_newton(nodes, runge(nodes))
s = spline(nodes, runge(nodes))
err_poly = np.max(np.abs(p(t) - runge(t)))
err_spline = np.max(np.abs(s(t) - runge(t)))
print(f"Runge, 30 equispaced nodes on [-10, 10]: max error polynomial {err_poly:.3e}, natural spline {err_spline:.3e}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, runge(t), "k", lw=2, label="f(x) = 1 / (1 + x²)")
ax.plot(t, p(t), label="degree-29 interpolating polynomial")
ax.plot(t, s(t), "--", label="natural cubic spline")
ax.scatter(nodes, runge(nodes), s=12, zorder=3, color="k")
ax.set_ylim(-0.5, 1.5)
ax.set_title("Runge's phenomenon: 30 equispaced nodes")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "runge_phenomenon.png", dpi=120)

# ---------------------------------------------------------------- 2. |x| with equispaced nodes
for n in (5, 10, 15, 20):
    x_nodes = np.array([-1 + 2 * k / n for k in range(n + 1)])
    P, _ = build_interpolator_newton(x_nodes, np.abs(x_nodes))
    tt = np.linspace(-1, 1, 2001)
    print(f"|x| with n = {n:2d}: max interpolation error {np.max(np.abs(P(tt) - np.abs(tt))):.3e}")

# ---------------------------------------------------------------- 3. ODE convergence
# Price adjustment model p'(t) = (120 - 8 p) / 4, p(0) = 20, exact p(t) = 5 e^{-2t} + 15.
def price_rhs(t, p):
    return (120 - 8 * p) / 4


def price_exact(t):
    return 5 * np.exp(-2 * t) + 15


ns = np.array([10, 20, 40, 80, 160, 320, 640])
h = 4 / ns
errors = {}
for name, method in [("Euler", euler), ("RK4", Runge4)]:
    errs = []
    for n in ns:
        xx, yy = method(price_rhs, 0, 4, n, 20)
        errs.append(np.max(np.abs(yy - price_exact(xx))))
    errors[name] = np.array(errs)
    slope = np.polyfit(np.log(h[2:]), np.log(errors[name][2:]), 1)[0]
    print(f"{name}: max error with n = 640 steps {errors[name][-1]:.3e}, observed order {slope:.2f}")

fig, ax = plt.subplots(figsize=(5, 4))
for name, marker in [("Euler", "o"), ("RK4", "s")]:
    ax.loglog(h, errors[name], marker=marker, label=name)
ax.loglog(h, 0.5 * h, "k:", lw=0.8, label="O(h)")
ax.loglog(h, 0.5 * h**4, "k--", lw=0.8, label="O(h⁴)")
ax.set_xlabel("step size h")
ax.set_ylabel("max error")
ax.set_title("Convergence on p' = (120 − 8p)/4")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "ode_convergence.png", dpi=120)

# ---------------------------------------------------------------- 4. Root finding
def f(x):
    return x**2 - 2


def df(x):
    return 2 * x


root = np.sqrt(2)
curves = {"bisection": [], "secant": [], "Newton": []}
for k in range(1, 41):
    curves["bisection"].append(abs(biseccion(f, 0, 2, k) - root))
for k in range(1, 8):
    curves["secant"].append(abs(secante(f, 0, 2, k) - root))
    curves["Newton"].append(abs(newton(f, df, 2.0, k) - root))
for name, errs in curves.items():
    first = next((i + 1 for i, e in enumerate(errs) if e < 1e-12), None)
    print(f"{name}: iterations to reach |x - sqrt(2)| < 1e-12: {first}")

fig, ax = plt.subplots(figsize=(5, 4))
for name, errs in curves.items():
    errs = np.maximum(errs, 1e-17)
    ax.semilogy(range(1, len(errs) + 1), errs, marker=".", label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("|x_k − √2|")
ax.set_title("Root of x² − 2 on [0, 2]")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "root_finding.png", dpi=120)
