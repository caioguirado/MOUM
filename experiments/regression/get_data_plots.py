import os
import json
import yaml
import pathlib
from tqdm import tqdm

from ...data.dataset import Dataset
from ...evaluation.uplift_curve import UpliftCurve
from ...experiments.experiment import Experiment

from ...evaluation.mo_regression import average_rmse

from ...models.enums import ModelEnum, MethodEnum

class GetPlots(Experiment):
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
        for result in results:
            result.plot_effects(save_filename=os.path.join(dir_name, f'{result.tradeoff_type}'))

    def run(self):

        # create dataset
        dimensions = self.config['dimensions']
        dataset_config = dimensions['dataset']
        td_types = dataset_config.pop('tradeoff_types')
        datasets = []
        for td in td_types:
            dataset_config['tradeoff_type'] = td
            dataset = Dataset(**dataset_config)
            datasets.append(dataset)

        self.save_results(datasets)