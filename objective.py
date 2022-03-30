import numpy as np

class Obj(object):
    def __init__(self):
        self.curr_best_y = np.inf
        self.curr_best_x = []

    def evaluate_true(self, x, func_name):
        f_x = None
        if func_name == 'Branin':
            f_x = Branin()
        elif func_name == 'Alpine1':
            f_x = Alpine1()
        elif func_name == 'Eggholder':
            f_x = Eggholder()
        elif func_name == 'Hartmann6':
            f_x = Hartmann6()
        elif func_name == 'Ackley2':
            f_x = Ackley2()
        elif func_name == 'Ackley10':
            f_x = Ackley10()
        elif func_name == 'Rosenbrock2':
            f_x = Rosenbrock2()
        elif func_name == 'Rosenbrock10':
            f_x = Rosenbrock10()
        elif func_name == 'BraninForrester':
            f_x = BraninForrester()
        elif func_name == 'Cosines':
            f_x = Cosines()
        elif func_name == 'GoldsteinPrice':
            f_x = GoldsteinPrice()
        elif func_name == 'SixHumpCamel':
            f_x = SixHumpCamel()

        f_eval = f_x.evaluate(x)
        f_eval = float(f_eval)
        if (f_eval < self.curr_best_y):
            self.curr_best_y = f_eval
            self.curr_best_x = x
        return np.array([f_eval])

    def evaluate(self, x, func_name):
        return self.evaluate_true(x, func_name)


class BraninForrester():
    def evaluate(self, X):
        input_dim = 2
        bounds = [(-5, 10), (0, 15)]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        xmin = [-3.689, 13.679]
        fmin = -16.64402
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s +5*x1
        return fval

class Cosines():

    bounds = [(5, 0), (5, 0)]
    xmin = [0.3125, 0.3125]
    fmin = -1.6
    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            g_x1 = (1.6*x1-0.5)**2
            g_x2 = (1.6*x2-0.5)**2
            r_x1 = 0.3*np.cos(3*np.pi*(1.6*x1-0.5))
            r_x2 = 0.3 * np.cos(3 * np.pi * (1.6 * x2 - 0.5))
            fval = -(1 - (g_x1-r_x1)-(g_x2-r_x2))
        return fval

class GoldsteinPrice():
    def evaluate(self, X):
        input_dim = 2
        bounds = [(-2, 2), (-2, 2)]
        xmin = [0, -1]
        fmin = 3
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:

            x1 = X[..., 0]
            x2 = X[..., 1]
            part1 = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
            part2 = (30 + (2 * x1 - 3 * x2) ** 2 * (
                        18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
            fval = part1 * part2
        return fval

class SixHumpCamel():
    def evaluate(self, X):
        input_dim = 2
        bounds = [(-3, 3), (-2, 2)]
        xmin = [[0.0898, -0.7126],[-0.0898,0.7126]]
        fmin = -1.0316
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:

            x1 = X[..., 0]
            x2 = X[..., 1]

            fval = (4-2.1*x1**2 + x1**4 /3)*x1**2 +x1*x2+(-4 + 4*x2**2)*x2**2
        return fval
# Branin objective function
class Branin():
    def evaluate(self, X):
        input_dim = 2
        bounds = [(-5, 10), (0, 15)]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        sd = 0

        xmin = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
        fmin = 0.397887
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return fval


# Alpine1 objective function
class Alpine1():
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]

    xmin = [(0, 0, 0, 0, 0)]
    fmin = 0

    def evaluate(self, X):
        input_dim = 5
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = np.abs(X * np.sin(X) + 0.1 * X).sum()
        return fval


# Egg Holder objective function
class Eggholder():
    def evaluate(self, X):
        input_dim = 2
        bounds = [(-512, 512), (-512, 512)]

        xmin = [(512, 404.2319)]
        fmin = -959.6407
        X = X.ravel()
        assert X.shape[0] == input_dim
        x1 = X[..., 0]
        x2 = X[..., 1]
        fval = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + 0.5 * x1 + 47))) - x1 * np.sin(
            np.sqrt(np.abs(x1 - (x2 + 47))))
        return fval


class Hartmann6():
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    xmin = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    fmin = -3.32237

    def evaluate(self, X):
        input_dim = 6
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = np.array([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            alp = np.array([1.0, 1.2, 3.0, 3.2])
            inner_sum = np.sum(A * (X - P) ** 2, axis=-1)
            fval = -(np.sum(alp * np.exp(-inner_sum), axis=-1))
        return fval


class Ackley2():
    bounds = [(-30, 30), (-30, 30)]

    xmin = [(0, 0)]
    fmin = 0

    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = -20 * np.exp(-0.02 * np.sqrt((np.sum(x1 ** 2 + x2 ** 2))/input_dim)) - np.exp(
                (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)) / 2) + 20 + np.exp(1)
        return fval


class Ackley10():
    bounds = [(-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30),
              (-30, 30)]

    xmin = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    fmin = 0

    def evaluate(self, X):
        input_dim = 10
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = -20 * np.exp(-0.02 * np.sqrt((np.sum(X ** 2))/input_dim)) - np.exp(
                np.sum(np.cos(2 * np.pi * X))/ input_dim)+ 20 + np.exp(1)
        return fval


class Rosenbrock2():
    bounds = [(-5, 10), (-5, 10)]

    xmin = [(1, 1)]
    fmin = 0

    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = 0
            for i in range(input_dim-1):
                fval = fval + 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2
        return fval


class Rosenbrock10():
    bounds = [(-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10)]
    xmin = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
    fmin = 0

    def evaluate(self, X):
        input_dim = 10
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = 0
            for i in range(input_dim-1):
                fval = fval + 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2
        return fval
