import os
import json
import pandas as pd

class Dataset():
    def __init__(self, config):
        self.data_path = config['data_path']
        self.dirc_name = self.data_path('/')[1]
        self.file_name = self.data_path('/')[2]
        self.standardize = config['standardize']

        with open(config['interval_path'], 'r') as file:
            self.intervals = json.load(file)['dirc_name']['file_name']

    def load_csv(data_path: str) -> pd.DataFrame:
        data = pd.read_csv(data_path)
        return data
    
    