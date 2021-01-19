from pathlib import Path
import requests
import torch

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip
import numpy as np

def get_examples():
        with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
                ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

        x_train, y_train, x_valid, y_valid = map(
                torch.tensor, (x_train, y_train, x_valid, y_valid)
        )

        x_train = x_train.numpy()
        y_train = y_train.numpy()
        x_valid = x_valid.numpy()
        y_valid = y_valid.numpy()

        return x_train, y_train, x_valid, y_valid