import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    rmse = np.sqrt(((pred - tar) ** 2).mean()) # TODO: Implement RMSE Calculation here...
    return rmse