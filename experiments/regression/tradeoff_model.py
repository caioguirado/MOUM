import os
import json
import yaml
import pathlib
import numpy as np
from tqdm import tqdm

from data.dataset import Dataset
from evaluation.uplift_curve import UpliftCurve
from experiments.experiment import Experiment

from evaluation.mo_regression import average_rmse

from models.enums import ModelEnum, MethodEnum

class TDModel(Experiment):
    def __init__(self, yaml_file) -> None:
        self.file_name = yaml_file.split('.') [0]
        self.current_file_path = pathlib.Path(__file__)
        config_path = os.path.join(self.current_file_path.parent.parent.resolve(), 'configs', yaml_file)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) 

    def save_results(self, results):
        with open(f'{self.current_file_path.parent.parent}/results/{self.file_name}.json', 'w') as file:
            json.dump(results, file, indent=4)

    def run(self):

        # create dataset
        dimensions = self.config['dimensions']
        dataset_config = dimensions['dataset']
        
        method = dimensions['method']
        tradeoff_types = dimensions['tradeoff_types']
        results = []
        for tradeoff_type in tqdm(tradeoff_types):

            dataset_config['tradeoff_type'] = tradeoff_type
            dataset = Dataset(**dataset_config)

            for model in dimensions['models']:
                print(f'evaluating....{tradeoff_type}_{model["enum"]}')
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
                    armse = average_rmse(tao_pred, dataset.Y_d_1[fold.test_idx, :] - dataset.Y_d_0[fold.test_idx, :])
                    aauuc = UpliftCurve(df=dataset.get_split(fold.test_idx).copy(), uplift=self.tao_pred, weights='mean').get_auuc()
                    cv_results.append({'armse': armse, 'aauuc': aauuc})

                results.append(dict(
                            tradeoff_type=tradeoff_type,
                            method=method,
                            model=model['enum'],
                            cv_results=cv_results
                        )
                )

        self.save_results(results)
                    
a = TDModel(yaml_file='experiment5.yaml')
a.run()