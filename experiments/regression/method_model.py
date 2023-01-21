import os
import json
import yaml
import shutil
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...data.dataset import Dataset
from ...experiments.experiment import Experiment

from ...evaluation.uplift_curve import UpliftCurve
from ...evaluation.mo_regression import average_rmse

from ...models.enums import ModelEnum, MethodEnum

class MethodModel(Experiment):
    def __init__(self, yaml_file) -> None:
        self.file_name = yaml_file.split('.') [0]
        self.current_file_path = pathlib.Path(__file__)
        config_path = os.path.join(self.current_file_path.parent.parent.resolve(), 'configs', yaml_file)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) 

    def save_results(self, results):
        dir_name = self.verify_file_structure(self.file_name)
        results = self.parse_results(results=results)
        df = self.get_df(results=results, row='method', column='model')

        with open(os.path.join(dir_name, f'{self.file_name}.json'), 'w') as file:
            json.dump(results, file, indent=4)
        
        with open(os.path.join(dir_name, 'table.tex'), 'w') as file:
            file.write(df.to_latex())

    def run(self):
        
        # create dataset
        dimensions = self.config['dimensions']
        dataset_config = dimensions['dataset']
        dataset = Dataset(**dataset_config)

        results = []
        for method in tqdm(dimensions['methods']):
            for model in dimensions['models']:
                print(f'evaluating....{method}_{model["enum"]}')
                method_class = MethodEnum[method].value
                model_class = ModelEnum[model['enum']].value
                method_obj = method_class(base_estimator=model_class, 
                                            base_estimator_kwargs=model.get('args', {}))
                print(model.get('args', {}))

                cv_results = []
                for fold in dataset.split():
                    method_obj.fit(X=dataset.X[fold.train_idx, :], 
                                    w=dataset.w[fold.train_idx, :], 
                                    Y=dataset.Y_obs[fold.train_idx, :])
                    tao_pred = method_obj.predict(X=dataset.X[fold.test_idx, :])
                    self.tao_pred = tao_pred
                    self.dataset = dataset
                    armse = average_rmse(tao_pred, dataset.Y_d_1[fold.test_idx, :] - dataset.Y_d_0[fold.test_idx, :])
                    aauuc = UpliftCurve(df=dataset.get_split(fold.test_idx).copy(), uplift=self.tao_pred, weights='mean').get_auuc()
                    cv_results.append({'armse': armse, 'aauuc': aauuc})

                results.append(dict(
                            method=method,
                            model=model['enum'],
                            cv_results=cv_results
                        )
                )

        print(results)
        self.save_results(results)