import os.path
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import numpy as np
import torch
import datasets
from Staging_model import StagingModel
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

data_names = ['cntomci', 'mcitoad']

dataset = data_names[0]
(x, y, time, label) = datasets.import_data(dataset=dataset)

save_path = 'result_adni_{}50'.format(dataset)
res_path = save_path + '/result'
if not os.path.exists(res_path):
    os.makedirs(res_path)
print('num features:', x.shape[1])
print('max times:', np.max(time))

batch_size = 32
epochs = 200
pre_train_epochs = 100
lr = 2e-4
pre_lr = 1e-5
nClusters = 2
input_dims = {
    'input_dim': x.shape[1] + 1,
    'x_dim': x.shape[1],
    'h_dim1': 200,
    'h_dim2': 100,
    'h_dim3': 50,
    'z_dim': 5,
    'nClusters': nClusters,
    'cs_dim1': 200,
    'cs_dim2': 200,
    'trans_dim1': 200,
    'trans_dim2': 100,
    'num_times': int(np.max(time) * 1.2),
}
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = StagingModel(x, y, time, label, input_dims=input_dims, nClusters=nClusters,
                     epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path)
model.fit(pre_train=True, pre_train_epochs=pre_train_epochs, pre_lr=pre_lr)


clusters = model.get_pred_cluster()


