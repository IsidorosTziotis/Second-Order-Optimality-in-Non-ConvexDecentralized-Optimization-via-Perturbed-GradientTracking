import os
import torch
import pandas as pd
import numpy as np
import dill as pickle


def save_dataset(dir, filename):
    M = np.zeros((943, 1682),dtype=float)
    df = pd.read_csv('u.data', delimiter='\t', header=None)
    for rows in range(df.shape[0]):
        M[df.iloc[rows, 0] - 1][df.iloc[rows, 1] - 1] = df.iloc[rows, 2]
    M = torch.from_numpy(M)

    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + os.path.sep + filename, 'wb') as f:
        torch.save(M, f)


def load_dataset(load_file):
    with open(load_file, 'rb') as f:
        M = torch.load(f)
    return M


if __name__ == "__main__":
    save_dataset("data", "M.p")
