import pandas as pd
import numpy as np

class UpliftCurve:
    def __init__(self, df, uplift, weights='mean'):
        self.df = df
        self.uplift = uplift
        if weights == 'mean':
            self.df['Y_obs'] = self.df.filter(regex=r"_obs").values.mean(axis=1, keepdims=True)
            self.uplift = self.uplift.mean(axis=1, keepdims=True)
            self.df['tao_combined'] = self.df.filter(regex=r"tao").values.mean(axis=1, keepdims=True)
        else:
            weights = np.array(weights).reshape(-1, 1)
            self.df['Y_obs'] = self.df.filter(regex=r"_obs").values.dot(weights)
            self.uplift = self.uplift.dot(weights)
            self.df['tao_combined'] = self.df.filter(regex=r"tao").values.dot(weights)
        
        self.df = self.df.assign(uplift=self.uplift)

    def get_group(self, w):
        return (
            self
            .df
            .query('w == @w')            
        )

    def get_group_sum(self, k, w, sort_column, sum_column='Y_obs', policy='uplift'):
        if policy == 'uplift':
            return (
                self
                .get_group(w)
                .sort_values(by=[sort_column], ascending=False)
                .head(k)
                [sum_column]
                .sum()
            )
        elif policy == 'random':
            return (
                self
                .get_group(w)
                # .sort_values(by=[column], ascending=False)
                .sample(frac=1) # shuffle
                .head(k)
                [sum_column]
                .sum()
            )

    def get_uplift(self, p, sort_column='uplift', sum_column='Y_obs', policy='uplift'):
        t_size = len(self.get_group(w=1))
        c_size = len(self.get_group(w=0))

        # eq. 9 in https://arxiv.org/abs/2002.05897
        r_t = self.get_group_sum(k=int(p*t_size), w=1, sort_column=sort_column, sum_column=sum_column, policy=policy)
        r_c = self.get_group_sum(k=int(p*c_size), w=0, sort_column=sort_column, sum_column=sum_column, policy=policy)

        return r_t/t_size - r_c/c_size

        
    def get_curve(self):
        ps = np.linspace(0, 1, 20).tolist()
        us = []
        taos = []
        random = []
        for p in ps:
            uplift_pred = self.get_uplift(p=p, sort_column='uplift', policy='uplift')
            us.append(uplift_pred)
            uplift_true = self.get_uplift(p=p, sort_column='tao_combined', policy='uplift')
            taos.append(uplift_true)
            uplift_random = self.get_uplift(p=p, sort_column='uplift', policy='random')
            random.append(uplift_random)

        return ps, us, taos, random

    def get_custom_rank_curve(self, column_mr, column_scalar_w):
        ps = np.linspace(0, 1, 20).tolist()
        mr_rank = []
        scalar_w = []
        random = []
        for p in ps:
            mr_rank_pred = self.get_uplift(p=p, sort_column=column_mr, policy='uplift')
            mr_rank.append(mr_rank_pred)
            scalar_w_pred = self.get_uplift(p=p, sort_column=column_scalar_w, policy='uplift')
            scalar_w.append(scalar_w_pred)
            uplift_random = self.get_uplift(p=p, sort_column='tao_combined', policy='random')
            random.append(uplift_random)

        return ps, mr_rank, scalar_w, random

    def get_auuc(self):
        return np.array(self.get_curve()[1]).sum() # subtract random curve?

# from ..data.dataset import Dataset
# import matplotlib.pyplot as plt
# dataset = Dataset(n_rows=1000, n_responses=2, X_dim=2, tradeoff_type='NON_LINEAR', prop_score=0.5)
# uc = UpliftCurve(df=dataset.df, uplift=np.random.rand(1000, 2))
# curve = uc.get_curve()
# # plt.plot(curve[0], curve[1], label='us')
# plt.plot(curve[0], curve[2], label='taos')
# plt.plot(curve[0], curve[3], label='random')
# plt.legend()
# plt.show()