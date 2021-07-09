"""
Using k-fold cross validation,
eval the data without train model.
"""
# loadbasic library
import pickle
import numpy as np
import random
from tqdm import tqdm

# load torch library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# load user-defined function
from tools import *

# keep reandom seed
seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


# check gpu
device = check_gpu()


# open data and stacking feature
X, y, fig_path, model_path, history_path = op_data_ensemble(True)


# k-fold
k = 5
train_list_x, train_list_y, cv_list_x, cv_list_y = k_fold_tensor(
    X, y, k=k, random_state=0
)


# create dataset
tr_set = []
va_set = []

for idx in range(len(train_list_x)):
    tr_set.append(Data(train_list_x[idx], train_list_y[idx]))

for idx in range(len(cv_list_x)):
    va_set.append(Data(cv_list_x[idx], cv_list_y[idx]))


# hypterparameter
batch_size = 32
dropout_rate = 0.1
neuron = 10
print()
print("batch_size:", batch_size)
print("dropout_rate:", dropout_rate)
print("neurons per layer:", neuron)
print()

# training
training_hist = []
for fold in tqdm(range(k)):

    model = Net(neuron=neuron, p=dropout_rate)
    model.to(device)
    print()
    print(model)
    print()
    summary(model, (0, 5))
    print()

    N_train = len(tr_set[fold])
    N_test = len(va_set[fold])
    print("[Fold]:", fold)
    print("Num of train samples:", N_train)
    print("Num of valid samples:", N_test)
    print()

    criterion = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(tr_set[fold], shuffle=True, batch_size=batch_size)

    validation_dataloader = DataLoader(
        va_set[fold], shuffle=False, batch_size=batch_size * 8
    )

    useful_stuff = eval_data_wiohout_train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        N_train=N_train,
        N_test=N_test,
        device=device,
        criterion=criterion,
    )

    training_hist.append(useful_stuff)
    print("*" * 50)


# save trainin_history
with open(history_path, "wb") as f:
    pickle.dump(training_hist, f)


# Calculate metrics
calc_metric_ensemble(training_hist, fig_path, train_metric=True, detail=False)
calc_metric_ensemble(training_hist, fig_path, train_metric=False, detail=False)


# plot learning curve
plot_lc_ensemble(training_hist, fig_path)
