"""
a sample d_covid19[0] has photo data d_covid19[0]['PA'], 14-length one hot vector label d_covid19['lab'],
and integer index d_covid19['idx']
A = posteroanterior,AP = anteroposterior, AP Supine = laying down
"""


import torchxrayvision as xrv
from tqdm import tqdm
import numpy as np

def dict_to_np():
    # Create data set:
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath="./covid-chestxray-dataset/images",
                                             csvpath="./covid-chestxray-dataset/metadata.csv")
    X = []
    Y = []
    for i in tqdm(range(len(d_covid19))):
        X.append(d_covid19[i]['PA'][0])
        Y.append(d_covid19[i]['lab'])
    np.save('labels', np.array(Y))
    np.save('CT_scans', np.array(X))

# Load data set:
# each X is a JPEG of varying size
X = np.load('CT_scans.npy', allow_pickle=True)
# 14-length multi-class one hot. Example: [0,0,1,0,1,0,0,0,1,0,0,0,0,0]. Multiple outcomes can be true for one sample
Y = np.load('labels.npy', allow_pickle=True)

# DATA PRE-PROCESSING


# DEFINE MODEL


# TRAIN


# TEST









