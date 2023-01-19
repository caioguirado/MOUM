import os
import yaml
import pathlib
from tqdm import tqdm

from data.dataset import Dataset
from experiments.experiment import Experiment

from evaluation.mo_regression import average_rmse

from models.enums import ModelEnum, MethodEnum

class MethodModel(Experiment):
    def __init__(self, yaml_file) -> None:
        self.path = ''
        config_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'configs', yaml_file)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) 

    def save_results(self):
        pass

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
                # print(model['args'])
                cv_results = []
                for fold in dataset.split():
                    method_obj.fit(X=dataset.X[fold.train_idx, :], 
                                    w=dataset.w[fold.train_idx, :], 
                                    Y=dataset.Y_obs[fold.train_idx, :])
                    tao_pred = method_obj.predict(X=dataset.X[fold.test_idx, :])
                    self.tao_pred = tao_pred
                    self.dataset = dataset
                    # evaluate
                    print(tao_pred.shape)
                    print(( dataset.Y_d_1[fold.test_idx, :] - dataset.Y_d_0[fold.test_idx, :]).shape)
                    armse = average_rmse(tao_pred, dataset.Y_d_1[fold.test_idx, :] - dataset.Y_d_0[fold.test_idx, :])
                    # add aAUUC
                    cv_results.append(armse)

                results.append(dict(
                            method=method,
                            model=model['enum'],
                            cv_results=cv_results
                        )
                )

        print(results)
                    
a = MethodModel(yaml_file='experiment1.yaml')
a.run()