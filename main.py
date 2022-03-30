import numpy as np

from DMEA_method import DMEA
import objective
import os

np.set_printoptions(precision=12, linewidth=500)


# test function
# func_name = ['Eggholder','Ackley2','Rosenbrock2','Alpine1','Branin', 'Hartmann6','BraninForrester','SixHumpCamel']

def get_bound(f_n):
    bounds = None
    if f_n == 'Branin':
        bounds = [[-5, 10], [0, 15]]
    elif f_n == 'Alpine1':
        bounds = [[-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10]]
    elif f_n == 'Eggholder':
        bounds = [[-512, 512], [-512, 512]]
    elif f_n == 'Hartmann6':
        bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    elif f_n == 'Ackley2':
        bounds = [[-30, 30], [-30, 30]]
    elif f_n == 'Rosenbrock2':
        bounds = [[-5, 10], [-5, 10]]
    elif f_n == 'BraninForrester':
        bounds = [[-5, 10], [0, 15], ]
    elif f_n == 'SixHumpCamel':
        bounds = [[-3, 3], [-2, 2]]
    dim = len(bounds)
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i in range(dim):
        lb[i] = bounds[i][0]
        ub[i] = bounds[i][1]

    return dim, lb, ub

def f(x):
    return obj_f.evaluate(x, f_n)[0]


if __name__ == '__main__':

    # test function
    f_n = 'Branin'

    dim, lb, ub = get_bound(f_n)
    num_init = 11 * dim - 1
    obj_f = objective.Obj()

    y_best_path = f'y_best/{f_n}'
    dbxy_path = f'dbxy/{f_n}'
    path = [y_best_path, dbxy_path]
    b = os.getcwd().replace('\\', '/')
    for pa in path:
        isExist = os.path.exists(b + '/' + pa)
        if not isExist:
            os.makedirs(b + '/' + pa)

    max_iteration = [45, 140]
    max_iter = 0
    if dim < 9:
        max_iter = max_iteration[0]
    else:
        max_iter = max_iteration[1]

    k = 4  # batch size
    eta = 0  # eta is a hyperparameters

    optimizer = DMEA(f, lb, ub, num_init, max_iter, k, mo_eval=7000, func_name=f_n, path=path, eta=eta)
    optimizer.init()
    optimizer.optimize()
