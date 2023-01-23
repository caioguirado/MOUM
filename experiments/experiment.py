import os
import shutil
import numpy as np
import pandas as pd
from abc import ABC, abstractclassmethod

class Experiment(ABC):

    @abstractclassmethod
    def save_results(self):
        pass

    @abstractclassmethod
    def run(self):
        pass

    def verify_file_structure(self, file_name):
        dir_name = f'{self.current_file_path.parent.parent}/results/{file_name}'
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        return dir_name

    def parse_results(self, results):
        for result in results:
            cv_results = result['cv_results']
            arsme = np.array([cvr['armse'] for cvr in cv_results])
            aauuc = np.array([cvr['aauuc'] for cvr in cv_results])
            result['armse_cv_mean'] = arsme.mean()
            result['armse_cv_std'] = arsme.std()
            result['aauuc_cv_mean'] = aauuc.mean()
            result['aauuc_cv_std'] = aauuc.std()
        
        return results

    def get_df(self, results, row, column, drop_cols=['cv_results']):
        df = (
            pd
            .DataFrame(results)
            .drop(columns=drop_cols)
            .round(4)
            .assign(
                armse=lambda x: x['armse_cv_mean'].astype(str) + '(' + x['armse_cv_std'].astype(str) + ')',
                aauuc=lambda x: x['aauuc_cv_mean'].astype(str) + '(' + x['aauuc_cv_std'].astype(str) + ')',
            )
            .pivot(index=row,
                        columns=column, 
                        values=['armse', 'aauuc'])
        )
        
        return df.T