import os

import pandas as pd

from .utils import DATA_NAME


def load_csv():
    with open(os.path.join("data", DATA_NAME), "r") as f:
        data = f.read()
    df = pd.read_csv(data)
    for i in range(len(df)):
        pass
        # questions = df["questions"][i].split(",")
        # categories = df["category"][i]
