import numpy as np
import pandas as pd
from itertools import combinations


def Preferred_select(batch_size, trust_vector, X_p, X_s, history_x):

    X_p = np.array(X_p)  # Pareto Front
    X_s = np.array(X_s)  # Pareto non-dominated solution

    # The value of the len(X_p)/10 closest to the maximum value on the three targets is regarded as a threshold
    bound = []
    for bb in range(len(trust_vector)):
        argsort = np.argsort(X_p[..., bb])
        bound.append(X_p[argsort[int(len(X_p) / 5)], bb])
    bound_comb = np.array(list(combinations(bound, 2)))

    # The 3D confidence vector is converted into 3 2D confidence vectors
    prefer_vec = np.array(trust_vector)

    prefer_vec_comb = np.array(list(combinations(prefer_vec, 2)))
    len_comb = len(prefer_vec_comb)

    # divide points into the different confidence intervals
    C_3 = []
    C_2 = []
    C_1 = []
    C_0 = []

    for k in range(len(X_p)):
        x_curr = X_p[k]
        x_curr_comb = np.array(list(combinations(x_curr, 2)))
        trust_condi = 0
        for jj in range(len_comb):
            prefer_vec_comb_2d = prefer_vec_comb[jj]
            bound_comb_2d = bound_comb[jj]
            x_curr_comb_2d = x_curr_comb[jj]
            if prefer_vec_comb_2d[0] == prefer_vec_comb_2d[1]:
                trust_condi = trust_condi + 1
            if prefer_vec_comb_2d[0] < prefer_vec_comb_2d[1]:
                if x_curr_comb_2d[0] <= bound_comb_2d[0]:
                    trust_condi = trust_condi + 1
            if prefer_vec_comb_2d[1] < prefer_vec_comb_2d[0]:
                if x_curr_comb_2d[1] <= bound_comb_2d[1]:
                    trust_condi = trust_condi + 1

        if trust_condi == 3:
            C_3.append(k)
        if trust_condi == 2:
            C_2.append(k)
        if trust_condi == 1:
            C_1.append(k)
        if trust_condi == 0:
            C_0.append(k)

    C_3 = np.array(C_3)
    C_2 = np.array(C_2)
    C_1 = np.array(C_1)
    C_0 = np.array(C_0)

    X_recommend = []

    # Round up 2/3 batch size, which is equal to high3
    assign_C3 = np.ceil(2 * batch_size / 3)

    history_x = np.array(history_x)

    total_assign = 0  # Total number of acquisitions

    assign_C3_stop = 0

    while True:
        if assign_C3_stop == assign_C3:
            break
        if C_3.shape[0] < 1:
            break
        if C_3.shape[0] >= 1:
            X_selct_index = random_sampling(1, C_3)
            x = np.array(X_s[X_selct_index])

            judge, history_x = judge_repeat(history_x, x.reshape(1, x.size))
            C_3 = np.delete(C_3, np.where(C_3 == X_selct_index))

            if judge == True:
                X_recommend.append(int(X_selct_index))
                assign_C3_stop = assign_C3_stop + 1
                total_assign = total_assign + 1

    batch_size_left = batch_size - total_assign

    assign_C2_stop = 0
    while True:
        if assign_C2_stop == batch_size_left:
            break
        if assign_C2_stop == assign_C3:
            break
        if C_2.shape[0] < 1:
            break
        if C_2.shape[0] >= 1:
            X_selct_index = random_sampling(1, C_2)
            x = np.array(X_s[X_selct_index])

            judge, history_x = judge_repeat(history_x, x.reshape(1, x.size))
            C_2 = np.delete(C_2, np.where(C_2 == X_selct_index))

            if judge == True:
                X_recommend.append(int(X_selct_index))
                assign_C2_stop = assign_C2_stop + 1
                total_assign = total_assign + 1

    batch_size_left = batch_size - total_assign

    assign_C1_stop = 0
    while True:
        if assign_C1_stop == batch_size_left:
            break
        if assign_C1_stop == assign_C3:
            break
        if C_1.shape[0] < 1:
            break
        if C_1.shape[0] >= 1:
            X_selct_index = random_sampling(1, C_1)
            x = np.array(X_s[X_selct_index])

            judge, history_x = judge_repeat(history_x, x.reshape(1, x.size))
            C_1 = np.delete(C_1, np.where(C_1 == X_selct_index))

            if judge == True:
                X_recommend.append(int(X_selct_index))
                assign_C1_stop = assign_C1_stop + 1
                total_assign = total_assign + 1

    batch_size_left = batch_size - total_assign

    assign_C0_stop = 0
    while True:
        if assign_C0_stop == batch_size_left:
            break
        if assign_C0_stop == assign_C3:
            break
        if C_0.shape[0] < 1:
            break
        if C_0.shape[0] >= 1:
            X_selct_index = random_sampling(1, C_0)
            x = np.array(X_s[X_selct_index])

            judge, history_x = judge_repeat(history_x, x.reshape(1, x.size))
            C_0 = np.delete(C_0, np.where(C_0 == X_selct_index))
            if judge == True:
                X_recommend.append(int(X_selct_index))
                assign_C0_stop = assign_C0_stop + 1
                total_assign = total_assign + 1

    batch_size_left = batch_size - total_assign
    if batch_size_left > 0:
        assign_C_all_stop = 0
        X_trust_high_total = np.concatenate((C_3, C_2, C_1, C_0), axis=-1)
        while True:
            if assign_C_all_stop == batch_size_left:
                break
            if X_trust_high_total.shape[0] < 1:
                break
            if X_trust_high_total.shape[0] >= 1:
                X_selct_index = random_sampling(1, X_trust_high_total)
                x = np.array(X_s[int(X_selct_index)])

                judge, history_x = judge_repeat(history_x, x.reshape(1, x.size))
                X_trust_high_total = np.delete(X_trust_high_total, np.where(X_trust_high_total == X_selct_index))
                if judge == True:
                    X_recommend.append(int(X_selct_index))
                    assign_C_all_stop = assign_C_all_stop + 1
                    total_assign = total_assign + 1

    return X_recommend


def judge_repeat(history_x, new_x):
    x_np = np.append(history_x, new_x, axis=0)

    len_hs_bef = np.array(x_np).shape[0]
    x_pd = pd.DataFrame()
    for i in range(x_np.shape[1]):
        x_pd.loc[:, f'hx{i}'] = x_np[..., i].flatten()
    subset_dbx = [f'hx{i}' for i in range(x_np.shape[1])]

    x_pd = x_pd.drop_duplicates(subset=subset_dbx, keep="first")

    x_np = x_pd.to_numpy()
    len_hs_aft = np.array(x_np).shape[0]
    judge = False

    if len_hs_aft == len_hs_bef:
        judge = True
    return judge, x_np


def random_sampling(batch_size, x_r):  # return self. B recommended samples
    sample_index = np.random.randint(0, len(x_r), size=batch_size)
    return x_r[sample_index]
