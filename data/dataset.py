import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from typing import List

from ..tradeoffs.enums import TradeoffEnum

class Fold:
    def __init__(self, fold_n, train_idx, test_idx) -> None:
        self.fold_n = fold_n
        self.train_idx = train_idx
        self.test_idx = test_idx

class Dataset:  
    def __init__(self, 
                    n_rows, 
                    X_dim, 
                    n_responses, 
                    tradeoff_type, 
                    prop_score=0.5, 
                    n_quantiles=5):

        self.n_rows = n_rows
        self.X_dim = X_dim
        self.n_responses = n_responses
        self.tradeoff_type = tradeoff_type
        self.prop_score = prop_score
        self.n_quantiles = n_quantiles

        self.X = np.random.uniform(0, 1, size=(self.n_rows, X_dim))
        self.w = np.random.binomial(1, self.prop_score, size=self.n_rows).reshape(-1, 1)

        # create multi-output Y with synthetic counterfactuals
        self.Y = self.create_Y()

        # create Y_obs
        even_idxs = [i for i in range(self.Y.shape[1]) if i%2==0]
        odd_idxs = [i for i in range(self.Y.shape[1]) if i%2!=0]
        self.Y_d_0 = self.Y[:, even_idxs]
        self.Y_d_1 = self.Y[:, odd_idxs]
        self.Y_obs = np.where(self.w == 1, self.Y_d_1, self.Y_d_0)
        self.tao = self.Y_d_1 - self.Y_d_0

        columns = (
            [f'X_{i}' for i in range(self.X.shape[1])] + 
            ['w'] +
            ['Y_{i}_0' for i in range(self.Y_d_0.shape[1])] +
            ['Y_{i}_1' for i in range(self.Y_d_1.shape[1])] +
            ['Y_{i}_obs' for i in range(self.Y_obs.shape[1])] +
            ['tao_{i}' for i in range(self.tao.shape[1])]
            )
        self.df = pd.DataFrame(np.concatenate([
            self.X,
            self.w,
            self.Y_d_0,
            self.Y_d_1,
            self.Y_obs,
            self.tao
        ], axis=1), columns=columns)

    def split(self, n_splits=5) -> List[Fold]:
        # create quantization
        quantiles = pd.qcut(self.Y_obs.mean(axis=1, keepdims=False), q=self.n_quantiles).astype('str')
        quantiles_w = np.concatenate([self.w, quantiles.reshape(-1, 1)], axis=1)
        string_encoding = (
            np.apply_along_axis(lambda x: ''.join(x), axis=1, arr=quantiles_w)
        )

        folds = []
        skf = StratifiedKFold(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(skf.split(self.X, string_encoding)):
            folds.append(Fold(fold_n=i, train_idx=train_index, test_idx=test_index))

        return folds

    def get_split(self, idx):
        return self.df.iloc[idx, :].copy()

    def create_Y(self):
        self.tradeoff = TradeoffEnum[self.tradeoff_type].value()
        return self.tradeoff.create_Y(self.X, self.n_responses)

    def plot_effects(self, save_filename=None):
        alpha=0.08
        nrows = self.n_responses + 1
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        for i in range(nrows):
            taos = []
            for j in range(ncols):
                if i == nrows-1:
                    im = axs[i, j].hexbin(x=self.X[:, 0], 
                                        y=self.X[:, 1], 
                                        C=self.Y[:, j+1]-self.Y[:, j], 
                                        gridsize=12
                    )
                    # im = axs[i, j].tricontourf(self.X[:, 0], 
                    #                         self.X[:, 1], 
                    #                         self.Y[:, j+1]-self.Y[:, j]
                    # )    
                    axs[i, j].set_xlabel('X_0')
                    axs[i, j].set_ylabel('X_1')                    
                    fig.colorbar(im, ax=axs[i, j], label=f'tao_{j}')
                    hex_values = im.get_array().reshape(-1, 1)
                    print(hex_values.shape)
                    taos.append(hex_values)
                    offsets = im.get_offsets()
                else:
                    # sns.regplot(x=self.X[:, i], y=self.Y[:, j], ax=axs[i, j], scatter=False, label=f'Y_{j}_0')
                    # sns.regplot(x=self.X[:, i], y=self.Y[:, j+1], ax=axs[i, j], scatter=False, label=f'Y_{j}_1')
                    axs[i, j].scatter(self.X[:, i], self.Y[:, j], alpha=alpha, label=f'Y_{j}_0')
                    axs[i, j].scatter(self.X[:, i], self.Y[:, j+1], alpha=alpha, label=f'Y_{j}_1')
                    axs[i, j].set_xlabel(f'X_{i}')
                    axs[i, j].set_ylabel(f'Y_{j}')
                    axs[i, j].legend()
        
        uplifts = np.concatenate(taos, axis=1)
        pp = np.where((uplifts[:, 0] > 0) & (uplifts[:, 1] > 0), 1, 0)
        mm = np.where((uplifts[:, 0] > 0) & (uplifts[:, 1] < 0), 2, 0)
        mm += np.where((uplifts[:, 0] < 0) & (uplifts[:, 1] > 0), 2, 0)
        nn = np.where((uplifts[:, 0] < 0) & (uplifts[:, 1] < 0), 3, 0)
        uplifts = pp + mm + nn

        plt.tight_layout()
        if save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()
            print(self.X[:, 0].shape)
            plt.figure(figsize=(12, 8))
            plt.hexbin(x=offsets[:, 0],
                        y=offsets[:, 1],
                        C=uplifts,
                        gridsize=12
            )
            plt.legend()
            plt.show()

if __name__ == "__main__":
    dataset = Dataset(n_rows=1000, 
                        X_dim=2, 
                        n_responses=2, 
                        tradeoff_type='NON_LINEAR', 
                        prop_score=0.5)

    dataset.plot_effects()