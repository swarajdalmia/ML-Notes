
import warnings
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np

def func():
    print(device_lib.list_local_devices())


if __name__ == "__main__":
    func()

