import pandas as pd
import numpy as np

class UpliftCurve:
    def __init__(self, df, uplift, weights='mean'):
        self.df = df
        self.uplift = uplift
        if weights == 'mean':
            self.df['Y_obs'] = self.df.copy().filter(regex=r"_obs").values.mean(axis=1, keepdims=True)
            self.uplift = self.uplift.mean(axis=1, keepdims=True)
        else:
            weights = np.array(weights).reshape(-1, 1)
            self.df['Y_obs'] = self.df.copy().filter(regex=r"_obs").values.dot(weights)
            self.uplift = self.uplift.dot(weights)
        
        self.df = self.df.assign(uplift=self.uplift)

    def get_group(self, w):
        return (
            self
            .df
            .query('w == @w')            
        )

    def get_group_sum(self, k, w):
        return (
            self
            .get_group(w)
            .sort_values(by=['uplift'], ascending=False)
            .head(k)
            ['Y_obs']
            .sum()
        )

    def get_uplift(self, p):
        t_size = len(self.get_group(w=1))
        c_size = len(self.get_group(w=0))

        # eq. 9 in https://arxiv.org/abs/2002.05897
        r_t = self.get_group_sum(k=int(p*t_size), w=1)
        r_c = self.get_group_sum(k=int(p*c_size), w=0)

        baseline = r_c * (t_size / c_size)
        # return (r_t - baseline) / baseline
        return r_t/t_size - r_c/c_size

    def get_curve(self):
        ps = np.linspace(0, 1, 20)
        us = []
        for p in ps:
            u = self.get_uplift(p=p)
            us.append(u)

        return ps, np.array(us)

    def get_auuc(self):
        return self.get_curve()[1].sum() # subtract random curve?