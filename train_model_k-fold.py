"""
K-fold cross validation.
"""
# load basic library
from tqdm import tqdm
import random
import numpy as np
import pickle

# load torch library
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

# load user-defined function
from tools import *

# keep reandom seed
seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

# check gpu
device = check_gpu()

# open data =====
k = 5  # k-fold
alldata, model_path, history_path, fig_path, data_index = op_shuffle_data(seed_val)

# k-fold
X_train = alldata["abstract"]
y_train = alldata["label"]
train_list_x, train_list_y, cv_list_x, cv_list_y = k_fold(
    X_train, y_train, k=k, random_state=seed_val
)

# tokenize DataFrame
tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", do_lower_case=False
)

# tokenize
input_ids_train_dict, attention_masks_train_dict, labels_train_dict = tokenizing(
    train_list_x, train_list_y, k, tokenizer
)
input_ids_cv_dict, attention_masks_cv_dict, labels_cv_dict = tokenizing(
    cv_list_x, cv_list_y, k, tokenizer, train=False
)

# Prepare torch dataset
tr_set = []
va_set = []
for idx in range(k):
    tr_set.append(
        TensorDataset(
            input_ids_train_dict["train" + str(idx)],
            attention_masks_train_dict["train" + str(idx)],
            labels_train_dict["train" + str(idx)],
        )
    )
    va_set.append(
        TensorDataset(
            input_ids_cv_dict["cv" + str(idx)],
            attention_masks_cv_dict["cv" + str(idx)],
            labels_cv_dict["cv" + str(idx)],
        )
    )

# hypterparameter
epochs = 2
batch_size = 4
print("epochs:", epochs)
print("batch_size:", batch_size)
print()

# training
training_hist = []
for fold in tqdm(range(k)):

    model = AutoModelForSequenceClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    N_train = len(tr_set[fold])
    N_test = len(va_set[fold])
    print("\n[Fold]:", fold)
    print("Num of train samples:", N_train)
    print("Num of valid samples:", N_test)
    print()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    )

    train_dataloader = DataLoader(tr_set[fold], shuffle=True, batch_size=batch_size)

    validation_dataloader = DataLoader(
        va_set[fold], shuffle=False, batch_size=batch_size
    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )

    useful_stuff = train_model(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        N_train=N_train,
        N_test=N_test,
        device=device,
        scheduler=scheduler,
        epochs=epochs,
    )

    training_hist.append(useful_stuff)
    print("*" * 50)

# calculate metric
calc_metric(training_hist, data_index, train_metric=True, detail=False)
calc_metric(training_hist, data_index, train_metric=False, detail=False)

# save model
torch.save(model.state_dict(), model_path)

# save trainin_history
with open(history_path, "wb") as f:
    pickle.dump(training_hist, f)

# plot learning curve
plot_lc(training_hist, fig_path, data_index)
