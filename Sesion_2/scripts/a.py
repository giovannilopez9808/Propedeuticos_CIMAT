from sklearn import linear_model, datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def grad_quadratic(theta, f_params):
    '''
    Gradiente de la funcion de costo 
           sum_i (theta@x[i]-y[i])**2
    '''
    X = f_params['X']
    y = f_params['y']
    err = theta[0]*X+theta[1]-y
    partial0 = err
    partial1 = X*partial0
    gradient = np.concatenate((partial1, partial0), axis=1)
    return np.sum(gradient, axis=1)


def grad_exp(theta, f_params):
    '''
    Gradiente de la funcion de costo 
           sum_i 1-exp(-k(theta@x[i]-y[i])**2)
    '''
    kappa = f_params['kappa']
    X = f_params['X']
    y = f_params['y']
    err = theta[0]*X[:, 0]+theta[1]*X[:, 1]-y
    err2 = err*np.exp(-kappa*err**2)
    partial0 = X[:, 0]*err2
    partial1 = X[:, 1]*err2
    gradient = np.concatenate((partial1, partial0), axis=1)
    print(np.mean(gradient, axis=0))
    return np.mean(gradient, axis=0)


def GD(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de gradiente

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   función que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso, 
                    nIter = gd_params[0] número de iteraciones
                    alpha = gd_params[1] tamaño de paso alpha

    f_params  :   lista de parametros para la funcion objetivo
                    kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                    X     = f_params['X'] Variable independiente
                    y     = f_params['y'] Variable dependiente                   

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                    Theta[-1] es el valor alcanzado en la ultima iteracion
    '''

    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    Theta = []
    for t in range(nIter):
        p = grad(theta, f_params=f_params)
        theta = theta - alpha*p
        Theta.append(theta)
    return np.array(Theta)


def NAG(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso acelerado de Nesterov

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso, 
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      eta   = gd_params['eta']  parametro de inercia (0,1]
    f_params  :   lista de parametros para la funcion objetivo, 
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente                   

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    eta = gd_params['eta']
    p = np.zeros(theta.shape)
    Theta = []

    for t in range(nIter):
        pre_theta = theta - 2.0*alpha*p
        g = grad(pre_theta, f_params=f_params)
        p = g + eta*p
        theta = theta - alpha*p
        Theta.append(theta)
    return np.array(Theta)


def ADAM(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de Gradiente Adaptable con Momentum(A DAM) 

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso, 
                      nIter    = gd_params['nIter'] número de iteraciones
                      alphaADA = gd_params['alphaADAM'] tamaño de paso alpha
                      eta1     = gd_params['eta1'] factor de momentum para la direccion 
                                 de descenso (0,1)
                      eta2     = gd_params['eta2'] factor de momentum para la el 
                                 tamaño de paso (0,1)
    f_params  :   lista de parametros para la funcion objetivo, 
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente                   

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    epsilon = 1e-8
    nIter = gd_params['nIter']
    alpha = gd_params['alphaADAM']
    eta1 = gd_params['eta1']
    eta2 = gd_params['eta2']
    p = np.zeros(theta.shape)
    v = 0.0
    Theta = []
    eta1_t = eta1
    eta2_t = eta2
    for t in range(nIter):
        g = grad(theta,
                 f_params)
        p = eta1*p + (1.0-eta1)*g
        v = eta2*v + (1.0-eta2)*(g**2)
        #p = p/(1.-eta1_t)
        #v = v/(1.-eta2_t)
        theta = theta - alpha * p / (np.sqrt(v)+epsilon)
        eta1_t *= eta1
        eta2_t *= eta2
        Theta.append(theta)
    return np.array(Theta)


def rosenbrock_function():
    x = np.linspace(-2.5, 2.5, 500)
    y = np.linspace(-2.5, 2.5, 500)
    z = (1-x)**2+200*(y-x*x)**2
    x = np.column_stack((x, y))
    z = np.expand_dims(z, axis=1)
    return x, z


# condición inicial
theta = 10*np.random.normal(size=2)

# parámetros del algoritmo
gd_params = {'alpha': 0.95,
             'alphaADADELTA': 0.7,
             'alphaADAM': 0.95,
             'nIter': 300,
             'batch_size': 100,
             'eta': 0.9,
             'eta1': 0.9,
             'eta2': 0.999}

# parámetros de la función objetivo
f_params = {'kappa': 0.01,
            'X': [],
            'y': []}
f_params["X"], f_params["y"] = rosenbrock_function()
ThetaGD = GD(theta=theta,
             grad=grad_exp,
             gd_params=gd_params,
             f_params=f_params)
print('Inicio:', theta, '-> Fin:', ThetaGD[-1, :])

ThetaNAG = NAG(theta=theta,
               grad=grad_exp,
               gd_params=gd_params,
               f_params=f_params)
print('Inicio:', theta, '-> Fin:', ThetaNAG[-1, :])

ThetaADAM = ADAM(theta=theta,
                 grad=grad_exp,
                 gd_params=gd_params,
                 f_params=f_params)
print('Inicio:', theta, '-> Fin:', ThetaADAM[-1, :])


mpl.rcParams['legend.fontsize'] = 14
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(projection='3d')
nIter = np.expand_dims(np.arange(ThetaGD.shape[0]), 1)
Tmax = 200
ax.plot(ThetaGD[:Tmax, 0],  ThetaGD[:Tmax, 1], nIter[:Tmax, 0], label='GD')
ax.plot(ThetaNAG[:Tmax, 0], ThetaNAG[:Tmax, 1], nIter[:Tmax, 0], label='NAG')
ax.plot(ThetaADAM[:Tmax, 0], ThetaADAM[:Tmax, 1],
        nIter[:Tmax, 0], label='ADAM')
ax.legend()
ax.set_title(r'Trayectorias los parámetros calculados con distintos algoritmos')
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_0$1')
ax.set_zlabel('Iteración')
ax.view_init(15)
plt.show()
