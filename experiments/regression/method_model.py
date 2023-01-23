import os
import json
import yaml
import shutil
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def save_figures(self, results, dir_name):
        # plot uplift curves
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        for result in results:
            plt.plot(result['uplift_curve'][0], result['uplift_curve'][1], 
                    label=f'{result["method"]}_{result["model"]}')
        # plt.plot(results[0]['uplift_curve'][0], results[0]['uplift_curve'][2], label='True uplift') # tao true
        plt.plot(results[0]['uplift_curve'][0], results[0]['uplift_curve'][-1], label='Random', linestyle='--', color='r') # random policy
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
        # plt.legend()
        # plt.tight_layout()
        plt.xlabel('Included population [p]')
        plt.ylabel('Uplift')
        plt.title('Uplift Curve')
        plt.savefig(os.path.join(dir_name, f'uplift_curve.png'))

    def save_results(self, results):
        dir_name = self.verify_file_structure(self.file_name)
        results = self.parse_results(results=results)
        df = self.get_df(results=results, row='method', column='model', drop_cols=['cv_results', 'uplift_curve'])

        self.save_figures(results=results, dir_name=dir_name)

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
                    armse = average_rmse(Y_true=dataset.tao[fold.test_idx, :], Y_pred=tao_pred, scale=False)
                    aauuc =  UpliftCurve(df=dataset.get_split(fold.test_idx).copy(), 
                                            uplift=tao_pred, weights='mean').get_auuc()
                    cv_results.append({'armse': armse, 'aauuc': aauuc})

                tao_pred_dataset = method_obj.fit_predict(X=dataset.X, 
                                                            w=dataset.w, 
                                                            Y=dataset.Y_obs)
                uc_plot = UpliftCurve(df=dataset.df.copy(), 
                                            uplift=tao_pred_dataset, weights='mean').get_curve()

                results.append(dict(
                            method=method,
                            model=model['enum'],
                            cv_results=cv_results,
                            uplift_curve=uc_plot
                        )
                )
        self.save_results(results)