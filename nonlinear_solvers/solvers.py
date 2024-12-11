"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """

    def g(x):
        """ return the iteration function g(x) for given f(x) and f'(x) """
        return x - f(x) / df(x)

    r = -1
    x_r = x_0
    x_r_plus_1 = g(x_r)

    while abs(x_r_plus_1 - x_r) > eps and r < max_its:
        r = r + 1
        x_r = x_r_plus_1
        x_r_plus_1 = g(x_r)
    if r == max_its:
        raise  ConvergenceError(f'max_its of {max_its} has been exceeded')
    return x_r_plus_1


def bisection(f, a, b, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    a : float
        The left end of the initial bisection interval.
    b : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """
    its = 0
    c = (a + b) / 2
    while abs(f(c)) > eps:
        # ValueError checked during while loop
        if f(a) * f(b) >= 0:
            raise ValueError(f'f(x_0) and f(x_1) are not of the same sign')
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
        its += 1
    # ConvergenceError checked after while loop terminates    
    if its >= max_its:
        raise  ConvergenceError(f'max_its of {max_its} has been exceeded')
    return c

def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    try:
        print("Trying Newton's Method")
        root_n = newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        print('Newton did not work')
        try:
            print("Trying Bisection method")
            root_b = bisection(f, x_0, x_1, eps, max_its_b)
        except ValueError:
            print("Bisection did not work")
        else:
            print('Bisection worked')
            return root_b    
    else:
        print('Newton worked')
        return root_n
