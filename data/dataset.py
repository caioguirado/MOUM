import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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


        self.pp_mask = (self.tao[:, 0] > 0) & (self.tao[:, 1] > 0)
        self.mm_mask = (self.tao[:, 0] > 0) & (self.tao[:, 1] < 0)
        self.mm_mask += (self.tao[:, 0] < 0) & (self.tao[:, 1] > 0)
        self.nn_mask = (self.tao[:, 0] < 0) & (self.tao[:, 1] < 0)

        # pp = np.where(self.pp_mask, 1, 0)
        # mm = np.where(self.mm_mask, 2, 0)
        # nn = np.where(self.nn_mask, 3, 0)
        # self.clusters = pp + mm + nn
        self.clusters = self.get_hex_clusters(values=self.tao)
        
        self.cmap = 'RdYlGn'
        self.norm = BoundaryNorm(np.arange(0.5, 4, 1), plt.cm.get_cmap(self.cmap).N)

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

    def hexbin_truncate(self, x):
        return stats.mode(x)[0].item()

    def get_hex_clusters(self, values):
        pp_mask = (values[:, 0] > 0) & (values[:, 1] > 0)
        mm_mask = (values[:, 0] > 0) & (values[:, 1] < 0)
        mm_mask += (values[:, 0] < 0) & (values[:, 1] > 0)
        nn_mask = (values[:, 0] < 0) & (values[:, 1] < 0)

        pp = np.where(pp_mask, 3, 0)
        mm = np.where(mm_mask, 2, 0)
        nn = np.where(nn_mask, 1, 0)

        return pp + mm + nn

    def plot_effects(self, save_filename=None):
        alpha=0.2
        nrows = self.n_responses + 1
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        for i in range(nrows):
            hex_taos = []
            for j in range(ncols):
                if i == nrows-1:
                    im = axs[i, j].hexbin(x=self.X[:, 0], 
                                        y=self.X[:, 1], 
                                        C=self.tao[:, j], 
                                        gridsize=12, 
                                        # cmap='coolwarm',
                                        cmap=self.cmap,
                                        # vmin=self.tao[:, j].min(), 
                                        # vmax=self.tao[:, j].max(), 
                                        # clim=(self.tao[:, j].min(), self.tao[:, j].max())
                    )
                    # im = axs[i, j].tricontourf(self.X[:, 0], 
                    #                         self.X[:, 1], 
                    #                         self.Y[:, j+1]-self.Y[:, j]
                    # )
                    axs[i, j].set_xlabel(r'$X_1$')
                    axs[i, j].set_ylabel(r'$X_2$')
                    fig.colorbar(im, ax=axs[i, j], label=fr'$\tau_{j}$')
                    hex_values = im.get_array().reshape(-1, 1)
                    hex_taos.append(hex_values)
                    offsets = im.get_offsets()
                else:
                    # sns.regplot(x=self.X[:, i], y=self.Y[:, j], ax=axs[i, j], scatter=False, label=f'Y_{j}_0')
                    # sns.regplot(x=self.X[:, i], y=self.Y[:, j+1], ax=axs[i, j], scatter=False, label=f'Y_{j}_1')
                    axs[i, j].scatter(self.X[:, i], self.Y[:, 2*j], alpha=alpha, label=fr'$Y_{j}^0$')
                    axs[i, j].scatter(self.X[:, i], self.Y[:, 2*j+1], alpha=alpha, label=fr'$Y_{j}^1$')
                    axs[i, j].set_xlabel(fr'$X_{i+1}$')
                    axs[i, j].set_ylabel(fr'$Y_{j}$')
                    axs[i, j].legend()

        plt.tight_layout()

        plt.savefig(f'{save_filename}.png')
    
        # scatter clusters
        plt.figure(figsize=(12, 8))
        plt.scatter(self.X[:, 0][self.pp_mask], self.X[:, 1][self.pp_mask], s=45, alpha=0.8, c='g', label='++')
        plt.scatter(self.X[:, 0][self.mm_mask], self.X[:, 1][self.mm_mask], s=45, alpha=0.8, c='b', label='+-')
        plt.scatter(self.X[:, 0][self.nn_mask], self.X[:, 1][self.nn_mask], s=45, alpha=0.8, c='r', label='--')
        plt.xlabel(r'$X_1$')
        plt.ylabel(r'$X_2$')
        plt.legend()
        plt.savefig(f'{save_filename}_clusters.png')

        # hexbin clusters
        plt.figure(figsize=(12, 8))
        plt.hexbin(x=offsets[:, 0],
                    y=offsets[:, 1],
                    # C=self.clusters,
                    C=self.get_hex_clusters(values=np.concatenate(hex_taos, axis=1)),
                    cmap=self.cmap,
                    norm=self.norm,
                    # reduce_C_function=self.hexbin_truncate,
                    gridsize=12
        )
        cbar = plt.colorbar(ticks=np.arange(1, 4, 1))
        cbar.ax.set_yticklabels(['--', '+-', '++'])
        plt.savefig(f'{save_filename}_hexclusters.png')


if __name__ == "__main__":
    dataset = Dataset(n_rows=5000, 
                        X_dim=2, 
                        n_responses=2, 
                        tradeoff_type='LINEAR', 
                        prop_score=0.5)

    dataset.plot_effects()