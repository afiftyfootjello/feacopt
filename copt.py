#!/bin/python


'''
given an eval function and starting guess, runs newtons method

works for multivariate, scalar valued functions
'''

from typing import Callable
import numpy as np
import scipy.optimize as scopt

def newton(fun: Callable, dfun: Callable, x0: np.ndarray, tol=1e-3, it_lim= 1000, save_iterations = False):

    evals = 0
    x0 = np.asarray(x0)

    assert(x0.ndim == 1)

    jaco = dfun(x0)
    evals += 1

    assert(jaco.shape[1] == x0.shape[0])

    invjaco = np.linalg.pinv(jaco)
    update = np.matmul(invjaco, fun(x0))
    evals += 1

    error = np.linalg.norm(update)

    xn = x0 - update.ravel()

    if save_iterations:
        history = [x0, xn]
    else:
        history = []

    it = 0
    while error > tol:
        jaco = dfun(xn)
        evals += 1
        invjaco = np.linalg.inv(jaco)
        update = np.matmul(invjaco, fun(xn))
        evals += 1
        xn1 = xn - update.ravel()
        error = np.linalg.norm(update)
        xn = xn1
        if save_iterations:
            history.append(xn)

        if it > it_lim:
            break
        it += 1

    return xn, history, evals


def grad_desc(fun: Callable, dfun: Callable, x0: np.ndarray, proj: Callable=None, tol=1e-3, it_lim= 1000, save_iterations = False):

    evals = 0
    x0 = np.asarray(x0)

    assert(x0.ndim == 1)

    # helper to handle numpy arrays
    def grad_vec(xval):
        return dfun(xval).ravel()

    # Find gradient at start
    grad = grad_vec(x0)

    # perform line search
    alpha,funevals,gradevals,_,_,_ = scopt.line_search(fun, grad_vec, x0, -grad)
    evals += 1 + funevals + gradevals

    if alpha is None:
        # If line search did not converge, take a leap of faith
        alpha = 0.1

    # step in descent direction
    xn = x0 + alpha * -grad

    if proj is not None:
        # project onto feasible set
        xn = proj(xn)

    # Save each iteration
    if save_iterations:
        history = [x0, xn]
    else:
        history = []

    error = np.linalg.norm(grad)
    it = 0
    while error > tol:

        # find gradient at current point
        grad = grad_vec(xn)

        # line search
        alpha,funevals,gradevals,_,_,_ = scopt.line_search(fun, grad_vec, xn, -grad)
        evals += 1 + funevals + gradevals

        if alpha is None:
            # If line search did not converge, take a leap of faith
            alpha = 0.1

        # step in descent direction
        xn1 = xn + alpha * -grad
        if proj is not None:
            # project onto feasible set
            xn1 = proj(xn1)

        # check if we are converged yet
        error = np.linalg.norm(xn1 - xn)

        xn = xn1
        if save_iterations:
            history.append(xn)

        if it > it_lim:
            break
        it += 1

    return xn, history, evals

def test_newton():
    from pytest import approx
    def rosen(x):
        return (2-x[0])**2 + 200*(x[1]-x[0]**2)**2

    def grad_rosen(x):
        f1 = -2*(2-x[0]) + 400*(x[1]-x[0]**2) * -2.0*x[0]
        f2 = 400*(x[1]-x[0]**2)
        return np.asarray([[f1],[f2]])

    def hess_rosen(x):
        f11 = 2 - 800 * x[1] + 800 * 3 * x[0]**2
        f12 = -800 * x[0]
        f21 = -800 * x[0]
        f22 = 400

        return np.asarray([[f11, f12], [f21, f22]])


    x0 = np.asarray([2.0,1.5])

    # local result
    res2,_ = newton(grad_rosen, hess_rosen, x0)

    expected = [2.0, 4.0]
    for i,x in enumerate(res2):
        assert(x == approx(expected[i]))

def test_grad_desc():
    # grad descent sucks at convergence on rosenbrock
    from pytest import approx
    def rosen(x):
        return (2-x[0])**2 + 200*(x[1]-x[0]**2)**2

    def grad_rosen(x):
        f1 = -2*(2-x[0]) + 400*(x[1]-x[0]**2) * -2.0*x[0]
        f2 = 400*(x[1]-x[0]**2)
        return np.asarray([[f1],[f2]])


    x0 = np.asarray([1.5,3.5])

    # local result
    res2,_ = grad_desc(rosen, grad_rosen, x0, it_lim=100000)

    expected = [2.0, 4.0]
    for i,x in enumerate(res2):
        print(x)
        assert(x == approx(expected[i], abs=1e-3))


def test_proj_grad_desc():
    # grad descent with a projection step
    from pytest import approx
    def obj(x):
        return (2-x[0])**2 + 200*(x[1] - x[0]**2)**2

    def grad(x):
        return [-2*(2-x[0]) + -800*x[0]*(x[1]-x[0]**2), 400*(x[1]-x[0]**2)]

    def proj(x):
        #project onto half space x0 <= 1
        return [min(x[0],1),x[1]]


    x0 = np.asarray([1.5,3.5])

    # local result
    res2,_ = grad_desc(rosen, grad_rosen, x0, proj=proj, it_lim=100000)

    #expected = [2.0, 4.0]
    for i,x in enumerate(res2):
        print(x)
    #    assert(x == approx(expected[i], abs=1e-3))
