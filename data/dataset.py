import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.tradeoffs import TradeoffEnum

class Dataset:
    # prop score, n_X_cols, n_responses, tradeoff type, 
    def __init__(self, 
                    n_rows, 
                    X_dim, 
                    n_responses, 
                    tradeoff_type, 
                    prop_score=0.5):
        self.n_rows = n_rows
        self.X_dim = X_dim
        self.n_responses = n_responses
        self.tradeoff_type = tradeoff_type
        self.prop_score = prop_score
        self.X = np.random.uniform(0, 1, size=(self.n_rows, X_dim))
        self.w = np.random.binomial(1, self.prop_score, size=self.n_rows).reshape(-1, 1)

        self.Y = self.create_Y()

    def create_Y(self):
        self.tradeoff = TradeoffEnum[self.tradeoff_type].value()
        return self.tradeoff.create_Y(self.X, self.n_responses)

    def plot_effects(self):
        nrows = self.n_responses + 1
        ncols = self.X_dim
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        for i in range(nrows):
            for j in range(ncols):
                if i == nrows-1:
                    im = axs[i, j].hexbin(x=self.X[:, 0], 
                                        y=self.X[:, 1], 
                                        C=self.Y[:, j+1]-self.Y[:, j], 
                                        gridsize=10
                    )
                    axs[i, j].set_xlabel('X_0')
                    axs[i, j].set_ylabel('X_1')                    
                    fig.colorbar(im, ax=axs[i, j], label=f'tao_{j}')
                else:
                    sns.regplot(x=self.X[:, i], y=self.Y[:, j], ax=axs[i, j], scatter=False, label=f'Y_{j}_0')
                    sns.regplot(x=self.X[:, i], y=self.Y[:, j+1], ax=axs[i, j], scatter=False, label=f'Y_{j}_1')
                    axs[i, j].scatter(self.X[:, i], self.Y[:, j], alpha=0.05)
                    axs[i, j].scatter(self.X[:, i], self.Y[:, j+1], alpha=0.05)
                    axs[i, j].set_xlabel(f'X_{i}')
                    axs[i, j].set_ylabel(f'Y_{j}')
                    axs[i, j].legend()
        
        plt.tight_layout()
        plt.show()
            
dataset = Dataset(n_rows=1000, X_dim=2, n_responses=2, tradeoff_type='HIGHLY_NON_LINEAR')
dataset.plot_effects()