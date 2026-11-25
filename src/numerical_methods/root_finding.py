"""Root finding: bisection, secant and Newton-Raphson."""
import numpy as np


def biseccion(func, a, b, n=3):
    
    c = (a+b)/2
    sign_a = np.sign(func(a))
    sign_b = np.sign(func(b))
    if sign_a == sign_b:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")
    if func(c) == 0 or n == 0:
        return c

    sign_c = np.sign(func(c))

    if sign_a == sign_c:
        return biseccion(func,c,b,n-1)
    
    else:
        return biseccion(func, a, c, n-1)


def secante(func, X_n, X_m, n = 3):    
    if n == 0:
        return X_m
    
    if func(X_m) == func(X_n):
        raise ValueError("División por cero en el método de la secante")
    
    X_new = X_m - (func(X_m) * (X_m - X_n))/ (func(X_m) - func(X_n))

    return secante(func, X_m, X_new, n - 1)


def newton(func, func_prima, X, n = 3):
    f_p = func_prima(X)
    if f_p == 0:
        raise ValueError("Division entre cero.")
    
    if n == 0:
        return X
    
    f = func(X)

    X_new = X - (f/f_p)

    return newton(func, func_prima, X_new, n - 1)
