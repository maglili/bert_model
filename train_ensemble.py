# loadbasic library
import pickle
import numpy as np
import random
from tqdm import tqdm

# load torch library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
X, y, fig_path, model_path, history_path = op_data_ensemble()


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
epochs = 150
batch_size = 32
lr = 1e-2
print("\nepochs:", epochs)
print("batch_size:", batch_size)
print("learning rate:", lr)
print()


# training
training_hist = []
for fold in tqdm(range(k)):

    model = Net(neuron=10, p=0.1)
    model.to(device)
    print(model)
    print()

    N_train = len(tr_set[fold])
    N_test = len(va_set[fold])
    print("[Fold]:", fold)
    print("Num of train samples:", N_train)
    print("Num of valid samples:", N_test)
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.5
    )

    train_dataloader = DataLoader(tr_set[fold], shuffle=True, batch_size=batch_size)

    validation_dataloader = DataLoader(
        va_set[fold], shuffle=False, batch_size=batch_size * 8
    )

    useful_stuff = train_ensemble(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        N_train=N_train,
        N_test=N_test,
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
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
