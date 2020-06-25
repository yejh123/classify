import os

class Config:
    def __init__(self):
        self.sample_data_dir = "../data/sample"
        self.test_data_dir = "../data/test"
        self.data_dir = "../data"
        self.prediction_dir = "../data/prediction"
        self.mode = 'sample'
        self.plot = False
        self.analysis = True
        self.fit_with_linear_regression = True

