import GPy
from GPyOpt.util.general import get_quantiles
import GPyOpt
import numpy as np
from math import pow, log, sqrt
import pandas as pd


# train_x : history x
# train_y : history y
# num_init : The number of initial sample points
# lsx : the collected x in last round
# lsy : the text function value corresponding to the collected x in last round
# P : Represents the historical confidence level of the acquisition functions in acquisition library
# domain : refer to the information of the variable of BayesianOptimization called by GPyOpt
# f : refer to the optimization objectives information of BayesianOptimization called by GPyOpt
# num_obj : The selected number of acquisition functions that build up multi-objective optimization.
# eta is a hyperparameters

class GP:
    def __init__(self, iter, train_x, train_y, exx, exy, k, num_init, lsx, lsy, P, f,
                 domain, model, eta=0.9):
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()

        self.mean = np.mean(exy)
        self.std = np.std(exy)
        self.num_train = exx.shape[0]

        self.lsx = lsx.copy()
        self.lsy = lsy.copy()
        self.exx = exx.copy()
        self.exy = exy.copy()

        self.dim = self.exx.shape[1]
        self.k = k
        self.num_init = num_init
        self.P = P
        self.eta = eta
        self.domain = domain
        self.f = f
        self.iter = iter
        self.m = model

        self.tau = np.min(train_y)
        self.burnin = 200
        self.n_samples = 10
        self.subsample_interval = 10
        self.sample()
        self.update()

        self.mean = np.mean(train_y)
        self.std = np.std(train_y)
        self.train_y = (train_y.copy() - self.mean) / self.std
        self.num_train = train_x.shape[0]
        kern = GPy.kern.Matern52(input_dim=self.dim, ARD=True)
        self.m = GPy.models.GPRegression(self.train_x, self.train_y, kern, noise_var=0)

    def update(self):

        tx = self.train_x.tolist()
        ty = self.train_y.ravel().tolist()
        lsy = self.lsy.ravel().tolist()
        self.lsbesty = np.min(self.exy)
        history_data = pd.DataFrame({'x': tx, 'y': ty})
        history_data.sort_values(by='y', ascending=True, inplace=True)
        history_data.reset_index(drop=True, inplace=True)

        self.rank = []
        self.hq_x = []
        self.phi_alpha = []
        alpha = 3
        for i in lsy:
            if len(history_data.index[history_data['y'] < i]) > 0:
                self.rank.append(history_data.index[history_data['y'] < i][-1] + 1)
            else:
                self.rank.append(0)

            if self.rank[-1] <= int(alpha):
                self.hq_x.append(1)
            else:
                self.hq_x.append(0)
        self.phii = self.hq_x
        self.rank = np.array(self.rank)
        self.hq_x = np.array(self.hq_x)

        # Stores information about the collection library
        self.acqusition_type = ['MPI', 'EI', 'LCB', 'LCB', 'LCB', 'LCB', 'LCB']
        self.hyperpara = [0.001, 0.001, [0.5, 0.5], [0.5, 0.05], [5, 0.1], [10, 0.1], [30, 0.1]]

        # LP_recommend_num is the regarding threshold of recommended number when using LP-alpha(x) acquisition functions
        # beta is equal to 0.5
        LP_recommend_num = int(0.5 * self.k) # k / 2
        current_P = []

        for i in range(len(self.acqusition_type)):
            if self.acqusition_type[i] == 'LCB':
                self.set_kappa(self.hyperpara[i][0], self.hyperpara[i][1])
                LP_checker = GPyOpt.methods.BayesianOptimization(f=self.f,
                                                                 domain=self.domain,
                                                                 acquisition_type=self.acqusition_type[i],
                                                                 normalize_Y=True,
                                                                 X=self.exx,
                                                                 Y=self.exy,
                                                                 evaluator_type='local_penalization',
                                                                 batch_size=LP_recommend_num,
                                                                 acquisition_weight=self.kappa)
            else:
                LP_checker = GPyOpt.methods.BayesianOptimization(f=self.f,
                                                                 domain=self.domain,
                                                                 acquisition_type=self.acqusition_type[i],
                                                                 normalize_Y=True,
                                                                 X=self.exx,
                                                                 Y=self.exy,
                                                                 evaluator_type='local_penalization',
                                                                 batch_size=LP_recommend_num,
                                                                 acquisition_jitter=self.hyperpara[i])

            LP_checker.run_optimization(max_iter=1)
            LP_acq_x = LP_checker.suggested_sample
            # acq_recommend_bound is the threshold at which the acquisition function recommends the sample points or not
            acq_recommend_bound = np.inf

            for x in LP_acq_x:
                pys, pss = self.predict(x)
                if self.acqusition_type[i] == 'LCB':
                    acq_calc = -self.LCB(pys, pss)
                elif self.acqusition_type[i] == 'MPI':
                    acq_calc = self.PI(pys, pss, self.hyperpara[i])
                else:
                    acq_calc = self.EI(pys, pss, self.hyperpara[i])
                if acq_calc < acq_recommend_bound:
                    acq_recommend_bound = acq_calc
            # acq_val is the value of the acquisition function
            # under the current acquisition function of the previous round of sampling points
            acq_val = []
            phi_alpha = []
            for x in self.lsx:
                pys, pss = self.predict(x)
                if self.acqusition_type[i] == 'LCB':
                    acq_calc = -self.LCB(pys, pss)
                elif self.acqusition_type[i] == 'MPI':
                    acq_calc = self.PI(pys, pss, self.hyperpara[i])
                else:
                    acq_calc = self.EI(pys, pss, self.hyperpara[i])
                acq_val.append(acq_calc)
                if acq_calc >= acq_recommend_bound - 1e-7:
                    phi_alpha.append(1)
                else:
                    phi_alpha.append(0)
            phi_alpha = np.array(phi_alpha)
            self.phi_alpha.append(phi_alpha.tolist())
            # cp is used to calculate the penalty value of current acquisition function
            cp = 0
            for i in range(self.lsx.shape[0]):
                cp += abs(self.hq_x[i] - phi_alpha[i]) * abs(self.lsbesty - lsy[i])
                if self.hq_x[i] == 1 and phi_alpha[i] == 1:
                    cp += lsy[i] - self.lsbesty
            current_P.append(cp)

        current_P = np.array(current_P)
        # self.CP is a comprehensive information that synthesizes the historical penalty value of acquisition functions
        # and the penalty value obtained by the previous round of sampling points,
        # and needs to be stored by calling the main function
        self.CP = self.eta * self.P + current_P
        df_CP = pd.DataFrame({'cp': self.CP.tolist()})
        # self.current_choice stores the subscripts of the num_obj acquisition functions selected in the acquisition
        # function library.
        # self.confidence returns the confidence level obtained by recombining the penalty value of the selected
        # acquisition functions in the current round, which is a one-dimensional ndarray
        # num_obj : The number of selected acquisition functions that build up multi-objective optimization
        # num_obj is equal to 3
        self.current_choice = df_CP.index[df_CP['cp'].rank(method='first') - 3 < 1e-3].tolist()
        self.T = []
        minn = np.min(self.CP)
        for i in self.current_choice:
            self.T.append(self.CP[i])
        self.T = np.array(self.T)
        self.T = (self.T + 1e-4) / np.sum((self.T + 1e-4))

    def sample(self):
        self.m.optimize_restarts(num_restarts=10)
        if not 0:
            self.s = np.array(np.array(self.m[:]))
            self.s = self.s.reshape(1, self.s.size)
            self.ms = np.array([self.m])
        else:
            hmc = GPy.inference.mcmc.HMC(self.m, stepsize=5e-2)
            s = hmc.sample(num_samples=self.n_samples * self.subsample_interval)
            self.s = s[0::self.subsample_interval]
            self.ms = []
            for i in range(self.s.shape[0]):
                samp_kern = GPy.kern.Matern52(input_dim=self.dim, ARD=True)
                samp_m = GPy.models.GPRegression(self.train_x, self.train_y, samp_kern)
                samp_m[:] = self.s[i]
                samp_m.parameters_changed()
                self.ms = np.append(self.ms, samp_m)

    def predict_sample(self, x, hyp_vec):
        self.m.kern.variance = hyp_vec[0]
        self.m.kern.lengthscale = hyp_vec[1:1 + self.dim]
        self.m.likelihood.variance = hyp_vec[1 + self.dim]
        py, ps2 = self.m.predict(x.reshape(1, x.size))
        py = self.mean + (py * self.std)
        ps2 = ps2 * (self.std ** 2)
        return py, ps2

    def set_kappa(self, upsilon, delta):
        num_train = self.num_train
        t = 1 + max(int((num_train - self.num_init) / self.k), 0)
        self.kappa = sqrt(
            upsilon * 2 * log(pow(t, 2.0 + self.dim / 2.0) * 3 * pow(np.pi, 2) / (3 * delta)))  # kappa of LCB

    def predict(self, x):
        num_samples = self.s.shape[0]
        pys = np.zeros((num_samples, 1))
        pss = np.zeros((num_samples, 1))
        for i in range(num_samples):
            m, v = self.ms[i].predict(x.reshape(1, x.size))
            pys[i] = m[0][0]
            pss[i] = v[0][0]
        pys = self.mean + (pys * self.std)
        pss = pss * (self.std ** 2)
        return pys, np.sqrt(pss)

    def LCB(self, pys, pss):
        num_samples = pys.shape[0]
        acq = 0
        for i in range(num_samples):
            y = pys[i]
            s = pss[i]
            lcb = y - self.kappa * s
            acq += lcb
        acq /= self.s.shape[0]
        return acq

    def EI(self, pys, pss, eps):
        num_samples = pys.shape[0]
        acq = 0
        for i in range(num_samples):
            y = pys[i]
            s = pss[i]
            phi, Phi, u = get_quantiles(eps, self.tau, y, s)
            f_acqu = s * (u * Phi + phi)
            acq += f_acqu
        acq /= self.s.shape[0]
        return acq

    def PI(self, pys, pss, eps):
        num_samples = pys.shape[0]
        acq = 0
        for i in range(num_samples):
            y = pys[i]
            s = pss[i]
            _, Phi, _ = get_quantiles(eps, self.tau, y, s)
            f_acqu = Phi
            acq += f_acqu
        acq /= self.s.shape[0]
        return acq

    def MACE_acq(self, x):
        pys, pss = self.predict(x)
        list = []
        for i in self.current_choice:
            if self.acqusition_type[i] == 'LCB':
                self.set_kappa(self.hyperpara[i][0], self.hyperpara[i][1])
                lcb = self.LCB(pys, pss)
                list.append([lcb, 0])
            elif self.acqusition_type[i] == 'MPI':
                pi = self.PI(pys, pss, self.hyperpara[i])
                list.append([pi, 1])
            else:
                ei = self.EI(pys, pss, self.hyperpara[i])
                list.append([ei, 1])
        return list
