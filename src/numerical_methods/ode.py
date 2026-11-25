"""Explicit one-step methods for y' = f(x, y): Euler and classical Runge-Kutta 4."""
import numpy as np


def euler(f, a, b, n, alpha):
    h = (b-a)/n
    xx = np.linspace(a, b, n+1)
    yy = np.zeros(n+1)

    yy[0] = alpha 

    for i in range(n):
        yy[i+1] = yy[i] + h*f(xx[i], yy[i])

    return xx, yy


def Runge4(f, a, b, n, alpha):
    h = (b-a)/n
    xx = np.linspace(a, b, n+1)
    yy = np.zeros(n+1)

    yy[0] = alpha

    for i in range(n):
        k1 = f(xx[i], yy[i])
        k2 = f(xx[i] + h/2, yy[i] + h/2 * k1)
        k3 = f(xx[i] + h/2, yy[i] + h/2 * k2)
        k4 = f(xx[i] + h, yy[i] + h*k3)
        yy[i+1] = yy[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return xx, yy
