"""
Training final model
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
k = 5  # k-fold cross validation
alldata, model_path, history_path, fig_path, data_index = op_shuffle_data(
    seed_val, final_model=True
)

# feature and target
sent_list = alldata["abstract"].to_list()
sent_label = alldata["label"].to_list()

# tokenize DataFrame
tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", do_lower_case=False
)

# tokenize
input_ids, attention_masks, labels = tokenizing_final(sent_list, sent_label, tokenizer)

# dataset
train_dataset = TensorDataset(input_ids, attention_masks, labels)

# hypterparameter
epochs = 2
batch_size = 4
print("epochs:", epochs)
print("batch_size:", batch_size)
print()

# init model
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

N_train = len(train_dataset)
print("\nNum of train samples:", N_train)
print()

# hyperparameters
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
)

# train the model
useful_stuff = train_model_final(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    N_train=N_train,
    device=device,
    scheduler=scheduler,
    epochs=epochs,
)

# calculate metric
calc_metric_final(useful_stuff, fig_path, data_index)

# save model
torch.save(model.state_dict(), model_path)

# save trainin_history
with open(history_path, "wb") as f:
    pickle.dump(useful_stuff, f)

# plot learning curve
plot_lc_final(useful_stuff, fig_path, data_index)
