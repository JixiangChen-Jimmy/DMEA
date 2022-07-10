'''
Code adapted from https://github.com/Alaya-in-Matrix/pyMACE
{pmlr-v80-lyu18a,
  title = 	 {Batch Bayesian Optimization via Multi-objective Acquisition Ensemble for Automated Analog Circuit Design},
  author =       {Lyu, Wenlong and Yang, Fan and Yan, Changhao and Zhou, Dian and Zeng, Xuan},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {3306--3314},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/lyu18a/lyu18a.pdf},
  url = 	 {https://proceedings.mlr.press/v80/lyu18a.html},
}
'''

from GP_model import GP
import numpy as np
from platypus import NSGAII, Problem, Real, Solution, InjectedPopulation, Archive
from scipy.optimize import fmin_l_bfgs_b
from smt.sampling_methods import LHS
import pandas as pd
import GPy
from prefer_select import Preferred_select


# eta is a hyperparameters
# lsx : the collected x in last round
# lsy : the test function value corresponding to the collected x in last round
class DMEA:
    def __init__(self, f, lb, ub, num_init, max_iter, k, path, mo_eval=7000, func_name=None, eta=0):
        """
        f: the objective function:
            input: D row vector
            output: scalar value
        lb: lower bound
        ub: upper bound
        num_init: number of initial random sampling
        max_iter: number of iterations
        B: batch size, the total number of function evaluations would be num_init + B * max_iter
        """
        self.f = f
        self.lb = lb.reshape(lb.size)
        self.ub = ub.reshape(ub.size)
        self.dim = self.lb.size
        self.num_init = num_init
        self.max_iter = max_iter
        self.k = k  # batch size
        self.mo_eval = mo_eval
        self.func_name = func_name
        self.eta = eta
        self.path = path
        self.ddbbxx = []

        domain = []
        for i in range(lb.size):
            gg = (self.lb[i], self.ub[i])
            domain.append({'name': f'x_{i + 1}', 'type': 'continuous', 'domain': gg})
        self.domain = domain

    def init(self):
        #  For initialization, the best self. B historical sampling points in the initial sampling are singled out
        #  as the historical sampling point of the recommended sampling points in previous round,
        #  so that the penalty information will be initialized at the first recommended point
        # The recommended sample point information for the previous round is stored in self.lsx, self.lsy
        # self.dbx stores all the X previously selected
        # self.dby stores all the Y previously selected
        self.dbx = np.zeros((self.num_init, self.dim))

        # Latin hypercube sampling
        xlimits = np.array([(self.lb[i], self.ub[i]) for i in range(self.dim)])
        samples = LHS(xlimits=xlimits, random_state=1)
        self.dbx = samples(self.num_init)
        self.ddbbxx = samples(self.num_init)

        self.dby = np.zeros((self.num_init, 1))
        self.best_y = np.inf

        self.min_y = []
        self.min_index = []

        for i in range(self.num_init):
            y = self.f(self.dbx[i])
            if y < self.best_y:
                self.best_y = y
                self.best_x = self.dbx[i]
            self.dby[i] = y

        # the best self. B points are taken out as the initial lsx, lxy
        dbx = self.dbx.tolist()
        dby = self.dby.tolist()
        db = pd.DataFrame({'dbx': dbx, 'dby': dby})
        self.lsy = np.array(db['dby'][db['dby'].rank(method='first') - self.k <= 1e-3].tolist()).reshape(-1, 1)
        self.lsx = np.array(db['dbx'][db['dby'].rank(method='first') - self.k <= 1e-3].tolist())
        self.exx = np.array(db.drop(db.index[[db['dby'].rank(method='first') - self.k <= 1e-3]])['dbx'].tolist())
        self.exy = np.array(db.drop(db.index[[db['dby'].rank(method='first') - self.k <= 1e-3]])['dby'].tolist())
        # Initialize the Gaussian model
        mean_ = np.mean(self.exy)
        std_ = np.std(self.exy)
        kern = GPy.kern.Matern52(input_dim=self.dim, ARD=True)

        self.m = GPy.models.GPRegression(self.exx, (self.exy - mean_) / std_, kern, noise_var=0)

    def gen_guess(self):
        num_guess = 1 + len(self.model.ms)
        guess_x = np.zeros((num_guess, self.dim))
        guess_x[0, :] = self.best_x

        def obj(x, m):
            m, _ = m.predict(x[None, :])
            return m

        def gobj(x, m):
            dmdx, _ = m.predictive_gradients(x[None, :])
            return dmdx

        bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        for i in range(1, num_guess):
            m = self.model.ms[i - 1]
            xx = self.best_x + np.random.randn(self.best_x.size).reshape(self.best_x.shape) * 1e-3

            print(np.random.randn(self.best_x.size))

            def mobj(x):
                return obj(x, m)

            def gmobj(x):
                return gobj(x, m)

            x, _, _ = fmin_l_bfgs_b(mobj, xx, gmobj, bounds=bounds)
            guess_x[i, :] = np.array(x)
        return guess_x

    def optimize(self):
        self.best_y = np.min(self.dby)
        self.P = np.zeros((1, 7)).ravel()
        for iter in range(self.max_iter):
            print('\n', self.func_name, f'{iter}/{self.max_iter}')
            self.model = GP(iter=iter, train_x=self.dbx, train_y=self.dby, exx=self.exx, exy=self.exy, k=self.k,
                            lsx=self.lsx, lsy=self.lsy, P=self.P, f=self.f, domain=self.domain,
                            num_init=self.num_init, model=self.m, eta=self.eta)
            self.P = self.model.CP
            if iter == 0:
                self.P = np.zeros((1, 7)).ravel()
            self.exx = self.dbx
            self.exy = self.dby
            self.m = self.model.m
            guess_x = self.gen_guess()
            print('Hello')
            num_guess = guess_x.shape[0]

            self.log = []

            # Build a multi-objective optimization problem
            def obj(x):
                val = []
                obj_list = self.model.MACE_acq(np.array([x]))
                for i in range(len(obj_list)):
                    self.log.append(obj_list[i][1])
                    if obj_list[i][1] == 1:
                        obj_list[i][0] = -1 * np.log(1e-40 + obj_list[i][0])
                    val.append(obj_list[i][0][0])
                return val

            # self.dim is the number of decision variables，
            # 3 is the number of multi-objective optimization problem，That is, the number of acquisition functions：
            # The acquisition functions are EI, LCB, and PI
            problem = Problem(self.dim, 3)

            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])

            init_s = [Solution(problem) for i in range(num_guess)]
            for i in range(num_guess):
                init_s[i].variables = [x for x in guess_x[i, :]]

            problem.function = obj
            gen = InjectedPopulation(init_s)
            arch = Archive()

            # to Use the NSGAII algorithm to calculate multi-objective optimization
            algorithm = NSGAII(problem, population=100, generator=gen, archive=arch)

            algorithm.run(self.mo_eval)

            if len(algorithm.result) > self.k:
                optimized = algorithm.result

            else:
                optimized = algorithm.population

            x_one_optimal = np.array(optimized[0].objectives)
            dimention = x_one_optimal.shape[0]
            trust_vector = self.model.T
            assert len(trust_vector) == dimention
            # X_pf Stored the pareto front of multi-objective acquisition functions optimized in this round
            # X_ps Stored the pareto optimal corresponding to X_pf in the round

            X_pf = np.array([np.array(optimized[i].objectives) for i in range(len(optimized))])
            X_ps = np.array([np.array(optimized[i].variables) for i in range(len(optimized))])
            # Use pandas to combine all X_pf and X_ps, and then delete the entire row
            # (including the column of X_pf and X_ps) depending on whether the X_ps is duplicated or not
            X_pf_ps = pd.DataFrame()

            for i in range(X_pf.shape[1]):
                X_pf_ps.loc[:, f'pf{i}'] = X_pf[..., i].flatten()

            for i in range(X_ps.shape[1]):
                X_pf_ps.loc[:, f'ps{i}'] = X_ps[..., i].flatten()

            subset_ps = [f'ps{i}' for i in range(X_ps.shape[1])]

            X_pf_ps.dropna(axis=0, how='any', inplace=True)
            X_pf_ps = X_pf_ps.drop_duplicates(subset=subset_ps, keep="first")

            # Remove duplicate elements from the X_pf
            X_pf = X_pf_ps.iloc[:, 0:X_pf.shape[1]]
            X_ps = X_pf_ps.iloc[:, -X_ps.shape[1]:]
            X_pf = np.array(X_pf)
            X_ps = np.array(X_ps)

            X_c = []

            if len(X_ps) > self.k:
                repeat_num = 0
                # Merge the three pareto optimals selected in this round and self.dbx,
                # and then check whether repeat or not and delete.
                #  comparing the newly merged self.dbx before and after the deletion,if the length of the array changes,
                #  the difference before and after array length is noted as repeat_num and assigned to
                #  the Preferred select strategy
                self_dbx_pd = pd.DataFrame()
                for i in range(self.dbx.shape[1]):
                    self_dbx_pd.loc[:, f'dbx{i}'] = self.dbx[..., i].flatten()
                subset_dbx = [f'dbx{i}' for i in range(self.dbx.shape[1])]

                self_dbx_pd = self_dbx_pd.drop_duplicates(subset=subset_dbx, keep="first")
                self.dbx = self_dbx_pd.to_numpy()

                for ii in range(dimention):
                    a = X_pf[:, ii]
                    list_a = a.tolist()
                    min_index = list_a.index(min(list_a))
                    x = np.array(X_ps[min_index])
                    self.dbx = np.append(self.dbx, x.reshape(1, x.size), axis=0)
                    len_dbx_bef = len(self.dbx)
                    dbx_pd = pd.DataFrame()
                    for i in range(self.dbx.shape[1]):
                        dbx_pd.loc[:, f'dbx{i}'] = self.dbx[..., i].flatten()
                    subset_dbx = [f'dbx{i}' for i in range(self.dbx.shape[1])]
                    dbx_pd = dbx_pd.drop_duplicates(subset=subset_dbx, keep="first")
                    self.dbx = dbx_pd.to_numpy()
                    len_dbx_after = len(self.dbx)
                    if len_dbx_after != len_dbx_bef:
                        repeat_num += 1
                    else:
                        X_c.append(min_index)

                # Preferred select strategy
                X_reco_pf = Preferred_select(self.k - dimention + repeat_num, trust_vector, X_pf, X_ps, self.dbx)

                for k in X_reco_pf:
                    X_c.append(k)
                prefer_x = X_reco_pf
                for i in prefer_x:
                    x = np.array(X_ps[i])
                    self.dbx = np.append(self.dbx, x.reshape(1, x.size), axis=0)

            elif len(X_ps) == self.k:

                for i in range(len(X_ps)):
                    X_c.append(i)
                for i in X_c:
                    x = np.array(X_ps[i])
                    self.dbx = np.append(self.dbx, x.reshape(1, x.size), axis=0)
            else:

                for i in range(len(X_ps)):
                    X_c.append(i)
                for i in range(self.k - len(X_ps)):
                    X_c.append(0)
                for i in X_c:
                    x = np.array(X_ps[i])
                    self.dbx = np.append(self.dbx, x.reshape(1, x.size), axis=0)
            lsx = []
            lsy = []
            # evaluate the selected B points
            for i in X_c:
                x = np.array(X_ps[i])
                y = self.f(x)  # evaluation

                lsx.append(x)
                lsy.append(y)
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x

                self.ddbbxx = np.append(self.ddbbxx, x.reshape(1, x.size), axis=0)
                self.dby = np.append(self.dby, y.reshape(1, 1), axis=0)
            self.dbx = self.ddbbxx

            self.lsx = np.array(lsx)
            self.lsy = np.array(lsy).reshape(-1, 1)

            # Save the Pareto front and solution in each iteration
            pf = np.array([s.objectives for s in optimized])
            ps = np.array([s.variables for s in optimized])
            self.pf = pf
            self.ps = ps

            np.savetxt(
                f'{self.path[1]}/dbx_Batchsize_{self.k}_{self.func_name}__eta_{self.eta}.txt',
                self.dbx)
            np.savetxt(
                f'{self.path[1]}/dby_Batchsize_{self.k}_{self.func_name}__eta_{self.eta}.txt',
                self.dby)

            # Save the smallest test function value in each iteration
            min_y_2 = self.dby.min()
            self.min_y.append(min_y_2)
            min_x_index = (np.where(self.dby == min_y_2)[0][0])
            min_index_2 = np.concatenate((np.array([min_y_2]), self.dbx[min_x_index]), axis=0)

            self.min_index.append(min_index_2)
            np.savetxt(f'{self.path[0]}/min_y_' + f'Batchsize{self.k}' + f'max_iter{self.max_iter}' + f'_{self.func_name}' + f'_eta_{self.eta}' + '.txt',
                       self.min_index, delimiter=',', fmt='%s')
