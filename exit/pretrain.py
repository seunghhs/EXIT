import yaml
import os
from exit import __root_dir__


with open(f'{__root_dir__}/config/tmp.yml', 'r') as file:
    config = yaml.safe_load(file)

# 경로를 동적으로 설정
config['dataset']['train_data_dir'] = os.path.join(__root_dir__, config['dataset']['train_data_dir'])