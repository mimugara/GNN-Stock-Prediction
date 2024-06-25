import pandas as pd
import os

DATA_KEYS = ["open","high","close","volume","low"]

class Underlying:
    def __init__(self):
        self.data = dict()
        self.load_data()

    def load_data(self):
        for key in DATA_KEYS:
            self.data[key] = pd.read_hdf("pv_ffill.h5", key=key)