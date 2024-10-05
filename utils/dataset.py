import os
import json
import pandas as pd
from torch.utils.data import Dataset

class OutlierDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]

    def load_csv(self) -> pd.DataFrame:
        """
        원본 csv파일 데이터프레임으로 불러오기.
        """
        data_path = self.config['data_path']
        data = pd.read_csv(data_path, usecols=['value'])
        return data
    
    def load_json(self) -> list:
        """
        비교적 이상치의 위험성이 적은 구간에 대한 정보 불러오기.
        """
        data_path = self.config['data_path']
        interval_path = self.config['interval_path']
        dirc_name = data_path.split('/')[1]
        file_name = data_path.split('/')[2]
        interval = json.load(interval_path)[dirc_name][file_name]
        return interval
    
    def split_by_interval(self,
                          data: pd.DataFrame,
                          intervals: list[list[int]]) -> list[pd.DataFrame]:
        """
        불러온 구간 정보로 원본 데이터프레임 분할, 분할한 데이터프레임을 리스트로 반환.
        """
        subsets = []
        for start, end in intervals:
            subset = data.iloc[start:end]
            subsets.append(subset)
        return subsets

    def slice_by_window(self, data: list[pd.DataFrame]) -> list[list[float]]:
        """
        분할된 데이터프레임별로 주어진 윈도우로 스텝 사이즈 만큼 이동하며 데이터 추출.
        만약에 주어진 구간 안에서 윈도우 설정이 불가능하면 해당 구간을 건너뛴다.
        """
        window_size = self.config['seq_size']
        step_size = self.config['step_size']
        pass

    def standardize(self, data: list[list[float]]) -> list[list[float]]:
        """
        입력 배열을 순서를 유지한채로 1차원으로 변환하고, 정규화를 진행.
        정규화한 값들을 다시 입력값과 같은 차원으로 변환.
        """
        pass

    def add_noise(self, data: list[list[float]]) -> list[list[float]]:
        """
        모든 값에 평균을 0, 표준편차를 1로 하는 정규분포로부터 샘플링된 노이즈를 추가한다.
        """
        pass

    def prepare_data(self, config):
        """
        """
        pass