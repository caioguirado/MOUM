import os
import json
import yaml
import pathlib
import numpy as np
import pandas as pd
import platypus as p
from tqdm import tqdm
import scipy.stats as ss
import matplotlib.pyplot as plt

from ...data.dataset import Dataset
from ...evaluation.uplift_curve import UpliftCurve
from ...experiments.experiment import Experiment

class MultirankScalarization(Experiment):
    def __init__(self, yaml_file) -> None:
        self.file_name = yaml_file.split('.') [0]
        self.current_file_path = pathlib.Path(__file__)
        config_path = os.path.join(self.current_file_path.parent.parent.resolve(), 'configs', yaml_file)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) 

    def save_figures(self, results, dir_name):
        pass

    def save_results(self, results):
        dir_name = self.verify_file_structure(self.file_name)
        nrows = 2
        ncols = 4
        n_splits = 10
        fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 9))
        fig.subplots_adjust(hspace=0.4)
        for j in range(ncols):
            # stacked bar non-cumulative
            stacked_bar_data = self.get_stacked_data(results[j]['values'], cumulative=False, n_splits=n_splits)
            stacked_bar_data.plot(kind='bar', stacked=True, ax=axis[0, j], legend=False, rot=0)
            axis[0, j].set_title(results[j]['name'], size=15)
            axis[0, j].margins(0,0)

            # stacked area cumulative
            stacked_area_data = self.get_stacked_data(results[j]['values'], cumulative=True, n_splits=n_splits)
            axis[1, j].stackplot(   
                np.arange(0, 1, 1/n_splits),
                stacked_area_data["pp"], 
                stacked_area_data["mm"],  
                stacked_area_data["nn"], 
                labels=['++','+-','--'])
            axis[1, j].margins(0,0)
            axis[1, j].set_title(results[j]['name'], size=15)
            
        axis[0, 0].set_xlabel('Decile')
        axis[0, 0].set_ylabel('Percentage of rank composition')
        axis[1, 0].set_xlabel('Proportion of population')
        axis[1, 0].set_ylabel('Percentage of rank composition')
        handles, labels = axis[1, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, f'rank_analysis.png'))


    def get_rank(self, ranks, scale=False, w=None):
        idx = range(len(ranks))
        if scale:
            ranks = ranks.dot(np.array([w, 1-w]).reshape(-1, 1))       
            ranks *= -1
        sorted_rank_idx = [*list(zip(ranks, idx))]
        sorted_rank_idx.sort(key=lambda x: x[0])
        mr_rank = np.array(sorted_rank_idx)[:, 1]

        return mr_rank

    def get_stacked_data(self, array, cumulative=False, n_splits=10):
        data = {
                'pp': [], 
                'mm': [],
                'nn': []
                }
        prev_chunk = []
        for chunk in np.split(array, n_splits):
            if cumulative:
                chunk = np.append(prev_chunk, chunk)
                prev_chunk = chunk.copy()
            data['pp'].append((chunk == 1).sum())
            data['mm'].append((chunk == 2).sum())
            data['nn'].append((chunk == 3).sum())

        data = pd.DataFrame(data, 
                index=range(1, 11)
            )
        data_perc = data.divide(data.sum(axis=1), axis=0)
        
        return data_perc

    def run(self):

        # create dataset
        dimensions = self.config['dimensions']
        dataset_config = dimensions['dataset']
        
        dataset = Dataset(**dataset_config)
        taos = dataset.tao
        
        # compute multirank
        problem = p.DTLZ2()
        solution_set = []
        for tao_1, tao_2 in taos:
            solution = p.Solution(problem)
            solution.objectives[:] = [-tao_1, -tao_2]
            solution_set.append(solution)

        results = []
        # MORanking
        p.nondominated_sort(solution_set)
        ranks = [sol.rank for sol in solution_set]
        mrank_idx = self.get_rank(ranks)
        ranked_clusters = dataset.clusters[mrank_idx]
        # self.get_stacked_plot(ranked_clusters, cumulative=True)
        results.append({'name': 'Non Dominated Sorting', 'values': ranked_clusters})

        # for w in 0-1 compute scalarization
        for w in np.arange(0, 1.1, 0.5):
            sw_rank = self.get_rank(taos, scale=True, w=w)
            ranked_clusters = dataset.clusters[sw_rank.astype(int)]
            # self.get_stacked_plot(ranked_clusters, cumulative=False)
            results.append({'name': fr'Scalarization w/ $w_1 = {w}$', 'values': ranked_clusters})

        self.save_results(results)   