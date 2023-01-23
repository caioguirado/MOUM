import os
import sys
import yaml
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from .experiments.enums import XPEnum

if __name__ == "__main__":

    # check if filename exists
    yaml_file = f'{sys.argv[1]}.yaml'
    current_file_path = pathlib.Path(__file__).parent
    config_path = (
        os.path.join(current_file_path.resolve(), 'experiments/configs', yaml_file)
    )
    print(yaml_file, config_path)
    if not pathlib.Path(config_path).is_file():
        raise Exception('File doesn\'t exist')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
    
    seed = config['global']['seed']
    np.random.seed(seed)
    plt.style.use('fivethirtyeight')
    
    xp_type = config['global']['xp_type']
    xp = XPEnum[xp_type].value(yaml_file=yaml_file)
    xp.run()