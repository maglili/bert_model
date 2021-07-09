"""
Using 5 sub-model to evaluate remaining data,
then save the result as input of ensemble model.
"""
# load basic library
import os
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# load user-defined function
from tools import *

# set gpu
device = check_gpu()

# open file
alldata, model_path, pickle_path = op_shuffle_data_eval(random_state=0)
# alldata, model_path, pickle_path = op_shuffle_data_eval_exp(random_state=0)
sent_list = alldata["abstract"].to_list()
sent_label = alldata["label"].to_list()

# tokenize DataFrame
tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", do_lower_case=False
)

# tokenize abstract
input_ids, attention_masks, labels = tokenizing_final(sent_list, sent_label, tokenizer)
# create dataset
train_dataset = TensorDataset(input_ids, attention_masks, labels)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print("\nModel in Training model:", model.training)

# hyper parameter
N_train = len(train_dataset)
print("\nNumber of train samples:", N_train)
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Eval whole dataset
feature_x = []
feature_y = []
with torch.no_grad():
    for batch in tqdm(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        logits = output[1]
        # _, yhat = torch.max(logits.data, 1)
        # feature_x.extend(yhat.tolist())
        prob_2dim = F.softmax(logits, dim=1)
        prob_1dim = prob_2dim[:, 0]
        feature_x.extend(prob_1dim.tolist())
        feature_y.extend(b_labels.tolist())

# save feature_X and feature_y as pickle
with open(pickle_path + "-X.pkl", "wb") as f:
    pickle.dump(feature_x, f)

feature_y_path = "./pickles/balance/1_dim_softmax-y.pkl"
if os.path.exists(feature_y_path):
    print("1_dim_softmax-y.pkl already exists.")
    with open(feature_y_path, "rb") as f:
        test = pickle.load(f)
    if test != feature_y:
        print("error!")
        quit()
    else:
        print("feature_y correct.")
else:
    with open(feature_y_path, "wb") as f:
        pickle.dump(feature_y, f)
