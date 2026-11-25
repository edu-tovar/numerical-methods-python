"""Natural cubic spline interpolation (method of moments)."""
import matplotlib.pyplot as plt
import numpy as np


def spline(xx, yy):
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    n_puntos = len(xx)
    n = n_puntos - 1  # número de intervalos
    
    A = np.zeros((n_puntos, n_puntos))
    d = np.zeros(n_puntos)

    A[0, 0] = 2    
    A[n, n] = 2

    # 2. Construir resto de la matriz 

    for j in range(1, n):
        h_j = xx[j] - xx[j-1]
        h_jplus1 = xx[j+1] - xx[j]
        
        lam_j = h_jplus1 / (h_j + h_jplus1)
        mu_j = 1 - lam_j
        dj = 6 * ((yy[j+1]-yy[j])/h_jplus1 - (yy[j]-yy[j-1])/h_j) / (h_j + h_jplus1)
        
        A[j, j] = 2
        A[j, j-1] = mu_j
        A[j, j+1] = lam_j
        d[j] = dj
        
    M = np.linalg.solve(A, d)  #momentos 



    def evaluar_spline(x):
        
        x = np.asarray(x, dtype=float)
        # 1. Localizar el intervalo j
        n = len(xx) - 1

        j = np.searchsorted(xx, x) - 1
        j = np.clip(j, 0, n - 1)
        
        h_jplus1 = xx[j+1] - xx[j]
        dx = x - xx[j] # Este es el (x - x_j)
        
        
        termino_3 = ((M[j+1] - M[j]) / (6 * h_jplus1)) * (dx**3)
        
        termino_2 = (M[j] / 2) * (dx**2)
        
        parentesis_lineal = ((yy[j+1] - yy[j]) / h_jplus1) - (((2 * M[j] + M[j+1]) * h_jplus1) / 6)
        termino_1 = parentesis_lineal * dx
        
        termino_0 = yy[j]
        
        return termino_3 + termino_2 + termino_1 + termino_0
            
    return evaluar_spline


def plot_spline(fun, x_nodes):
    
    y_nodes = fun(x_nodes)      #Evaluamos los puntos en la funcion real

    fun_spline = spline(x_nodes, y_nodes)       #Obtenemos la funcion de evaluacion para el Spline

    
    xs = np.linspace(min(x_nodes),max(x_nodes), 1000)       #Limites del plot
    

    #Conseguimos los puntos a plotear
    y_spline = fun_spline(xs)
    y_real = fun(xs)


    #PLOT
    plt.figure()
    plt.plot(xs, y_real, label='Funcion real')
    plt.plot(xs, y_spline, label='SPLINE')
    plt.scatter(x_nodes, y_nodes)
    plt.legend()
    plt.grid()
    plt.show()
