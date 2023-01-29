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

        ###
        # results = self.parse_results(results=results)
        # df = self.get_df(results=results, row='y_dim', column='model', drop_cols=['cv_results'])

        # self.save_figures(results=results, dir_name=dir_name)

        with open(os.path.join(dir_name, f'{self.file_name}.json'), 'w') as file:
            json.dump(results, file, indent=4)
        
        # with open(os.path.join(dir_name, 'table.tex'), 'w') as file:
        #     file.write(df.to_latex())

    def get_rank(self, ranks, scale=False, w=None):
        idx = range(len(ranks))
        if scale:
            ranks = ranks.dot(np.array([w, 1-w]).reshape(-1, 1))            
        sorted_rank_idx = [*list(zip(ranks, idx))]
        sorted_rank_idx.sort(key=lambda x: x[0])
        mr_rank = np.array(sorted_rank_idx)[:, 1]

        return mr_rank

    def get_stacked_plot(self, array, cumulative=False):
        n_splits = 10
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
                index=range(0, n_splits)
            )
        
        data_perc = data.divide(data.sum(axis=1), axis=0)

        plt.stackplot(
            np.arange(0, 1, 1/n_splits),  
            data_perc["pp"], 
            data_perc["mm"],  
            data_perc["nn"], 
            labels=['++','+-','--'])
        plt.legend(loc='upper left')
        plt.margins(0,0)
        plt.show()

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

        p.nondominated_sort(solution_set)
        ranks = [sol.rank for sol in solution_set]
        mrank_idx = self.get_rank(ranks)
        
        ranked_clusters = dataset.clusters[mrank_idx]
        # print(ranked_clusters[1500:2000])
        self.get_stacked_plot(ranked_clusters, cumulative=True)
        raise Exception('End MORanking')
        # for w in 0-1 compute scalarization
        results = []
        sw_baseline_rank = self.get_rank(taos, scale=True, w=0.5)
        for w in np.arange(0, 1.1, 0.1):
            sw_rank = self.get_rank(taos, scale=True, w=w)
            sw_baseline_kendalltau = ss.kendalltau(sw_rank, y=sw_baseline_rank)
            mr_baseline_kendalltau = ss.kendalltau(sw_rank, y=mr_rank)

            uc_plot = UpliftCurve(df=dataset.df.assign(mr=sw_baseline_rank, 
                                                        sw=sw_rank).copy(), 
                                    uplift=taos, weights='mean').get_custom_rank_curve(column_mr='mr', column_scalar_w='sw')
            results.append(dict(
                w=w,
                sw_baseline_kendalltau=sw_baseline_kendalltau,
                mr_baseline_kendalltau=mr_baseline_kendalltau,
                uc_plot=uc_plot
            ))

        # plot mr+mean, w=0.1, w=0.5, w=0.9 uplift curves
        # after, also check when varying w, if (--) cohort always stay behind

        self.save_results(results)   