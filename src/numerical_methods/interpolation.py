"""Polynomial interpolation in Newton form (divided differences)."""
import numpy as np
import matplotlib.pyplot as plt


def newton_formula(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
 
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
 
    if x_nodes.size == 0:
        return np.array([], dtype=float)
 
    n = x_nodes.size
    coeffs = y_nodes.copy()
 
    # coeffs[k] = f[x0, x1, ..., xk]
    for i in range(1, n):
        numerator = coeffs[i:n] - coeffs[i - 1:n - 1]
        denominator = x_nodes[i:n] - x_nodes[:n - i]
        coeffs[i:n] = numerator / denominator
 
    return coeffs


def polynomial_newton(x: np.ndarray, x_nodes: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evalúa el polinomio de Newton usando multiplicación anidada (tipo Horner para Newton).
 
    Parámetros
    ----------
    x : np.ndarray o float
        Puntos donde evaluar.
    x_nodes : np.ndarray
        Nodos usados para construir el polinomio.
    coeffs : np.ndarray
        Coeficientes de Newton (diferencias divididas): c0..cn.
 
    Devuelve
    --------
    np.ndarray
        Valores P(x).
    """
    x = np.asarray(x, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
 
    # P(x) = c0 + (x-x0)[c1 + (x-x1)[c2 + ...]]
    result = np.zeros_like(x, dtype=float)
    for k in range(coeffs.size - 1, -1, -1):
        result = coeffs[k] + (x - x_nodes[k]) * result
    return result


def build_interpolator_newton(x_nodes: np.ndarray, y_values: np.ndarray):
    """
    Devuelve una función p(x) que evalúa el polinomio interpolador, construido con Newton.
 
    Uso:
        p, coeffs = build_interpolator_newton(x_nodes, y_values)
        y = p(x)
    """
    coeffs = newton_formula(x_nodes, y_values)
 
    def p(x):
        return polynomial_newton(x, x_nodes, coeffs)
 
    return p, coeffs


def plot_function_and_interpolant(f, x_nodes: np.ndarray, y_nodes: np.ndarray, p, interval: tuple[float, float]) -> None:
    """
    Dibuja f(x) y el polinomio interpolador p(x), además de los nodos.
 
    Args:
        f: función original
        x_nodes, y_nodes (np.ndarray): nodos y valores (y_nodes = f(x_nodes))
        p: polinomio interpolador
        interval (tuple): intervalo de dibujo
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
 
    a, b = interval
 
    # Número de puntos para pintar las curvas
    num_points = 1000
    x_plot = np.linspace(a, b, num_points)
    y_f = f(x_plot)
    y_p = p(x_plot)
   
    title = "Función y polinomio interpolador"
    plt.figure()
    plt.plot(x_plot, y_f, label="f(x)")
    plt.plot(x_plot, y_p, label="P_n(x)")
    plt.scatter(x_nodes, y_nodes, label="Nodos", zorder=3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()
