# loadbasic library
import pickle
import random
import csv
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)

# load torch library
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch

# keep reandom seed
seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


def train_model(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    N_train,
    N_test,
    device,
    scheduler,
    epochs=4,
):
    """
    train the BERT model.

    Data definition:
        Training: metric that model in training model on train set.
        train: metric that model in eval model on train set.
        valid: metric that model in eval model on valid set.

    """

    useful_stuff = {
        "training_loss": [],
        "training_acc": [],
        "training_auc": [],
        "training_metric": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_metric": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_auc": [],
        "valid_metric": [],
    }

    for epoch in range(epochs):
        # Training========================================
        model.train()

        correct = 0
        training_loss = 0
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []

        for batch in train_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output[0]
            logits = output[1]
            loss.backward()
            _, yhat = torch.max(logits.data, 1)
            correct += (yhat == b_labels).sum().item()
            training_loss += loss.item()

            # calc metric
            TP += ((yhat == 1) & (b_labels == 1)).sum().item()
            FP += ((yhat == 1) & (b_labels == 0)).sum().item()
            TN += ((yhat == 0) & (b_labels == 0)).sum().item()
            FN += ((yhat == 0) & (b_labels == 1)).sum().item()

            # calc auc
            auc_y.extend(b_labels.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        acc_ = correct / N_train
        loss_ = training_loss / len(train_dataloader)

        useful_stuff["training_loss"].append(loss_)
        useful_stuff["training_acc"].append(acc_)
        useful_stuff["training_metric"].append((TP, FP, TN, FN))
        useful_stuff["training_auc"].append(auc)

        print("training loss: {0:.2f}".format(loss_))
        print("training acc: {0:.2f}".format(acc_))
        print("-" * 10)

        # train metirc ===================================
        model.eval()

        correct = 0
        train_loss = 0
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []

        for batch in train_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels
                )
                loss = output[0]
                logits = output[1]

            # Accumulate the validation loss.
            train_loss += loss.item()
            _, yhat = torch.max(logits.data, 1)
            correct += (yhat == b_labels).sum().item()

            # calc metric
            TP += ((yhat == 1) & (b_labels == 1)).sum().item()
            FP += ((yhat == 1) & (b_labels == 0)).sum().item()
            TN += ((yhat == 0) & (b_labels == 0)).sum().item()
            FN += ((yhat == 0) & (b_labels == 1)).sum().item()

            # calc auc
            auc_y.extend(b_labels.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        acc_ = correct / N_train
        loss_ = train_loss / len(train_dataloader)

        useful_stuff["train_loss"].append(loss_)
        useful_stuff["train_acc"].append(acc_)
        useful_stuff["train_metric"].append((TP, FP, TN, FN))
        useful_stuff["train_auc"].append(auc)

        print("train loss: {0:.2f}".format(loss_))
        print("train acc: {0:.2f}".format(acc_))
        print("-" * 10)

        # Validation========================================
        model.eval()

        correct = 0
        cv_loss = 0
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels
                )
                loss = output[0]
                logits = output[1]

            # Accumulate the validation loss.
            cv_loss += loss.item()

            _, yhat = torch.max(logits.data, 1)
            correct += (yhat == b_labels).sum().item()

            # calc metric
            TP += ((yhat == 1) & (b_labels == 1)).sum().item()
            FP += ((yhat == 1) & (b_labels == 0)).sum().item()
            TN += ((yhat == 0) & (b_labels == 0)).sum().item()
            FN += ((yhat == 0) & (b_labels == 1)).sum().item()

            # calc auc
            auc_y.extend(b_labels.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        acc_ = correct / N_test
        loss_ = cv_loss / len(validation_dataloader)

        useful_stuff["valid_loss"].append(loss_)
        useful_stuff["valid_acc"].append(acc_)
        useful_stuff["valid_metric"].append((TP, FP, TN, FN))
        useful_stuff["valid_auc"].append(auc)

        print("valid loss: {0:.2f}".format(loss_))
        print("valid acc: {0:.2f}".format(acc_))

        print("=" * 20)

    return useful_stuff


def op_shuffle_data(random_state, final_model=False):
    """
    Combine positive and negative data,
    return it as a DadaFrame.
    Args:
        random_state (int): set shuffle random seed
        final_model (bool): whether training the final model
    """

    # data path
    data_index = str(input("Dataset index (0~4): "))
    pos_path = "./data/train.csv"
    neg_path = "./data/balance/neg_train/neg_train-" + data_index + ".csv"

    # set path
    if not final_model:
        model_path = "./model/balance/kfold/neg_train-" + data_index + ".pt"
        history_path = "./history/balance/kfold/neg_train-" + data_index + ".pkl"
        fig_path = "./pics/balance/kfold/"
    else:
        model_path = "./model/balance/final/neg_train-" + data_index + ".pt"
        history_path = "./history/balance/final/neg_train-" + data_index + ".pkl"
        fig_path = "./pics/balance/final/"

    print("pos_path:", pos_path)
    print("neg_path:", neg_path)
    print("model_path:", model_path)
    print("history_path:", history_path)
    print("fig_path:", fig_path)
    print("-" * 20)

    # positive data
    df_pos = pd.read_csv(pos_path, encoding="utf-8", dtype=str).fillna("")
    df_pos = df_pos[["abstract"]]
    df_pos = df_pos.assign(label=1)

    # negative data
    try:
        df_neg = pd.read_csv(neg_path, encoding="utf-8", dtype=str).fillna("")
    except:
        print("Error opening files! Please check path.")
        quit()
    df_neg = df_neg[["abstract"]]
    df_neg = df_neg.assign(label=0)

    alldata = pd.concat([df_pos, df_neg], ignore_index=True)
    alldata = shuffle(alldata, random_state=random_state)

    return alldata, model_path, history_path, fig_path, data_index


def k_fold(X_train, y_train, k, random_state):
    """
    K-Fold Cross Validation.
    Input:
        X_train(DataFrame): abstract
        y_train(DataFrame): label
    Output:
        train_list_x(list): list of list of abstract
        train_list_y(list): list of list of label
        cv_list_x(list): list of list of abstract
        cv_list_y(list): list of list of label
    """
    skf = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=random_state
    )  # init class

    train_list_x = []
    train_list_y = []
    cv_list_x = []
    cv_list_y = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        # train_index
        train_list_x.append(X_train.iloc[train_index].to_list())
        train_list_y.append(y_train.iloc[train_index].to_list())

        # test_index
        cv_list_x.append(X_train.iloc[test_index].to_list())
        cv_list_y.append(y_train.iloc[test_index].to_list())

    return train_list_x, train_list_y, cv_list_x, cv_list_y


def check_gpu():
    """
    Check if GPU avaliable.
    """
    print()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    print()

    return device


def tokenizing(train_list_x, train_list_y, k, tokenizer, train=True):
    """
    Tokenize abstracts and return data as Bert model input tensor.
    """
    # save k-fold data
    input_ids_dict = {}
    attention_masks_dict = {}
    labels_dict = {}

    for idx in range(k):

        sent_list = train_list_x[idx]
        sent_label = train_list_y[idx]

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids_list = []
        attention_masks_list = []

        # For every sentence...
        for sent in sent_list:
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
                truncation=True,
            )

            # Add the encoded sentence to the list.
            input_ids_list.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks_list.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids_list = torch.cat(input_ids_list, dim=0)
        attention_masks_list = torch.cat(attention_masks_list, dim=0)
        labels = torch.tensor(sent_label)

        if train:
            input_ids_dict["train" + str(idx)] = input_ids_list
            attention_masks_dict["train" + str(idx)] = attention_masks_list
            labels_dict["train" + str(idx)] = labels
        else:
            input_ids_dict["cv" + str(idx)] = input_ids_list
            attention_masks_dict["cv" + str(idx)] = attention_masks_list
            labels_dict["cv" + str(idx)] = labels

    return input_ids_dict, attention_masks_dict, labels_dict


def calc_metric(training_history, data_index, train_metric=True, detail=False):
    """
    Calculate metric.
    """
    # init
    ACC = []
    LOSS = []
    RECALL = []
    SPECIFICITY = []
    PRECISION = []
    NPV = []
    F1 = []
    MCC = []
    AUC = []

    for i in range(len(training_history)):

        if train_metric:
            (TP, FP, TN, FN) = training_history[i]["train_metric"][-1]
            auc = training_history[i]["train_auc"][-1]
            loss = training_history[i]["train_loss"][-1]
            path = "./pics/balance/kfold/train-" + data_index + ".txt"
        else:
            (TP, FP, TN, FN) = training_history[i]["valid_metric"][-1]
            auc = training_history[i]["valid_auc"][-1]
            loss = training_history[i]["valid_loss"][-1]
            path = "./pics/balance/kfold/valid-" + data_index + ".txt"

        acc = (TP + TN) / (TP + FP + TN + FN)

        try:
            recall = TP / (TP + FN)  # 召回率是在所有正樣本當中，能夠預測多少正樣本的比例
        except:
            recall = 0

        try:
            specificity = TN / (TN + FP)  # 特異度是在所有負樣本當中，能夠預測多少負樣本的比例
        except:
            specificity = 0

        try:
            precision = TP / (TP + FP)  # 準確率為在所有預測為正樣本中，有多少為正樣本
        except:
            precision = 0

        try:
            npv = TN / (TN + FN)  # npv為在所有預測為正樣本中，有多少為正樣本
        except:
            npv = 0

        try:
            f1 = (2 * recall * precision) / (recall + precision)  # F1-score則是兩者的調和平均數
        except:
            f1 = 0

        try:
            mcc = (TP * TN - FP * FN) / np.sqrt(
                ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            )
        except:
            mcc = 0

        if detail:
            print("[ fold", i, "]", "(Total sample: {})".format(TP + FP + TN + FN))
            print("TP:", TP)
            print("FP:", FP)
            print("TN:", TN)
            print("FN:", FN)
            print()
            print("acc:", acc)
            print("loss:", loss)
            print("recall:", recall)
            print("specificity:", specificity)
            print("precision:", precision)
            print("npv:", npv)
            print("f1:", f1)
            print("mcc:", mcc)
            print("auc:", auc)
            print("=" * 40)

        ACC.append(acc)
        LOSS.append(loss)
        RECALL.append(recall)
        SPECIFICITY.append(specificity)
        PRECISION.append(precision)
        NPV.append(npv)
        F1.append(f1)
        MCC.append(mcc)
        AUC.append(auc)

    if train_metric:
        print("\n[Training average]\n")
    else:
        print("\n[valid average]\n")
    print("ACC: {:.2}".format((np.mean(ACC))))
    print("LOSS: {:.2}".format(np.mean(LOSS)))
    print()
    print("Recall: {:.2}".format(np.mean(RECALL)))
    print("Specificity: {:.2}".format(np.mean(SPECIFICITY)))
    print("Precision: {:.2}".format(np.mean(PRECISION)))
    print("NPV: {:.2}".format(np.mean(NPV)))
    print()
    print("F1: {:.2}".format(np.mean(F1)))
    print("MCC: {:.2}".format(np.mean(MCC)))
    print("AUC: {:.2}".format(np.mean(AUC)))
    print()

    # save result
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        if train_metric:
            writer.writerow(["[Train average]"])
        else:
            writer.writerow(["[valid average]"])
        writer.writerow(["ACC: {:.2}".format((np.mean(ACC)))])
        writer.writerow(["LOSS: {:.2}".format(np.mean(LOSS))])
        writer.writerow(["Recall: {:.2}".format(np.mean(RECALL))])
        writer.writerow(["Specificity: {:.2}".format(np.mean(SPECIFICITY))])
        writer.writerow(["Precision: {:.2}".format(np.mean(PRECISION))])
        writer.writerow(["NPV: {:.2}".format(np.mean(NPV))])
        writer.writerow(["F1: {:.2}".format(np.mean(F1))])
        writer.writerow(["MCC: {:.2}".format(np.mean(MCC))])
        writer.writerow(["AUC: {:.2}".format(np.mean(AUC))])


def calc_avg(training_history):
    """
    Plot learning curve
    """

    a1 = a2 = a3 = a4 = []  # init

    for i in range(len(training_history)):
        if i == 0:
            a1 = np.array(training_history[0]["train_loss"].copy())
            a2 = np.array(training_history[0]["valid_loss"].copy())
            a3 = np.array(training_history[0]["train_acc"].copy())
            a4 = np.array(training_history[0]["valid_acc"].copy())
            continue
        a1 = a1 + np.array(training_history[i]["train_loss"])
        a2 = a2 + np.array(training_history[i]["valid_loss"])
        a3 = a3 + np.array(training_history[i]["train_acc"])
        a4 = a4 + np.array(training_history[i]["valid_acc"])

    a1 /= len(training_history)
    a2 /= len(training_history)
    a3 /= len(training_history)
    a4 /= len(training_history)

    a1 = a1.tolist()
    a2 = a2.tolist()
    a3 = a3.tolist()
    a4 = a4.tolist()

    return a1, a2, a3, a4


def plot_lc(training_history, fig_path, data_index):

    a1, a2, a3, a4 = calc_avg(training_history)

    # color
    tr_color = ["#2ff5f2", "#2ff5e8", "#2ff5c0", "#2fbdf5", "#2f99f5"]
    val_color = ["#f5952f", "#f5ac2f", "#f5c02f", "#f5d72f", "#f5ee2f"]

    # loss
    for idx, color in enumerate(tr_color):  # train
        plt.plot(
            training_history[idx]["train_loss"],
            "--",
            alpha=0.4,
            label="train" + str(idx),
            color=color,
        )
    plt.plot(a1, label="average training")

    for idx, color in enumerate(val_color):  # valid
        plt.plot(
            training_history[idx]["valid_loss"],
            "--",
            alpha=0.4,
            label="valid" + str(idx),
            color=color,
        )
    plt.plot(a2, label="average valid")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.title("training / valid loss vs iterations")
    plt.grid()
    plt.savefig(fig_path + "set-" + data_index + "-loss" + ".png")
    plt.close()

    # acc
    for idx, color in enumerate(tr_color):  # train
        plt.plot(
            training_history[idx]["train_acc"],
            "--",
            alpha=0.4,
            label="train" + str(idx),
            color=color,
        )
    plt.plot(a3, label="average training")

    for idx, color in enumerate(val_color):  # valid
        plt.plot(
            training_history[idx]["valid_acc"],
            "--",
            alpha=0.4,
            label="valid" + str(idx),
            color=color,
        )
    plt.plot(a4, label="average valid")
    plt.ylabel("acc")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    plt.legend()
    plt.title("training / valid acc vs iterations")
    plt.grid()
    plt.savefig(fig_path + "set-" + data_index + "-acc" + ".png")
    plt.close()


def tokenizing_final(sent_list, sent_label, tokenizer):
    """
    Tokenize abstracts.
    """
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sent_list:

        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(sent_label)

    return input_ids, attention_masks, labels


def train_model_final(
    model, train_dataloader, optimizer, N_train, device, scheduler, epochs=4
):
    """
    Training the final model.
    """
    useful_stuff = {
        "training_loss": [],
        "training_acc": [],
        "training_metric": [],
        "training_auc": [],
        "train_loss": [],
        "train_acc": [],
        "train_metric": [],
        "train_auc": [],
    }

    for epoch in range(epochs):
        # Training========================================
        model.train()

        correct = 0
        training_loss = 0
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []

        for batch in train_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output[0]
            logits = output[1]
            loss.backward()
            _, yhat = torch.max(logits.data, 1)
            correct += (yhat == b_labels).sum().item()
            training_loss += loss.item()

            # calc metric
            TP += ((yhat == 1) & (b_labels == 1)).sum().item()
            FP += ((yhat == 1) & (b_labels == 0)).sum().item()
            TN += ((yhat == 0) & (b_labels == 0)).sum().item()
            FN += ((yhat == 0) & (b_labels == 1)).sum().item()

            # calc auc
            auc_y.extend(b_labels.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        useful_stuff["training_loss"].append(training_loss / len(train_dataloader))
        useful_stuff["training_acc"].append(correct / N_train)
        useful_stuff["training_metric"].append((TP, FP, TN, FN))
        useful_stuff["training_auc"].append(auc)

        print("training loss: {0:.2f}".format(training_loss / len(train_dataloader)))
        print("training acc: {0:.2f}".format(correct / N_train))
        print("-" * 10)

        # train========================================
        model.eval()
        with torch.no_grad():

            correct = 0
            training_loss = 0
            TP = FP = TN = FN = 0
            auc_y = []
            auc_yhat = []

            for batch in train_dataloader:

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                output = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels
                )
                loss = output[0]
                logits = output[1]
                _, yhat = torch.max(logits.data, 1)
                correct += (yhat == b_labels).sum().item()
                training_loss += loss.item()

                # calc metric
                TP += ((yhat == 1) & (b_labels == 1)).sum().item()
                FP += ((yhat == 1) & (b_labels == 0)).sum().item()
                TN += ((yhat == 0) & (b_labels == 0)).sum().item()
                FN += ((yhat == 0) & (b_labels == 1)).sum().item()

                # calc auc
                auc_y.extend(b_labels.cpu().detach().numpy())
                auc_yhat.extend(yhat.cpu().detach().numpy())

            fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
            auc = metrics.auc(fpr, tpr)
            useful_stuff["train_loss"].append(training_loss / len(train_dataloader))
            useful_stuff["train_acc"].append(correct / N_train)
            useful_stuff["train_metric"].append((TP, FP, TN, FN))
            useful_stuff["train_auc"].append(auc)

            print("train loss: {0:.2f}".format(training_loss / len(train_dataloader)))
            print("train acc: {0:.2f}".format(correct / N_train))
            print()
        print("=" * 20)
    print("*" * 40)
    return useful_stuff


def calc_metric_final(useful_stuff, fig_path, data_index):
    """
    calculate metrics.
    args:
        useful_stuff(dict): infomation dict, obtaining acc, loss, auc, ...
        fig_path(str): path to store figures
        data_index(int): what dataset is
    """
    path = fig_path + "final-" + data_index + ".txt"
    (TP, FP, TN, FN) = useful_stuff["train_metric"][-1]

    acc = (TP + TN) / (TP + FP + TN + FN)
    loss = useful_stuff["train_loss"][-1]

    recall = TP / (TP + FN)  # 召回率是在所有正樣本當中，能夠預測多少正樣本的比例
    specificity = TN / (TN + FP)  # 特異度是在所有負樣本當中，能夠預測多少負樣本的比例
    precision = TP / (TP + FP)  # 準確率為在所有預測為正樣本中，有多少為正樣本
    npv = TN / (TN + FN)  # npv為在所有預測為正樣本中，有多少為正樣本

    f1 = (2 * recall * precision) / (recall + precision)  # F1-score則是兩者的調和平均數
    mcc = (TP * TN - FP * FN) / np.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    auc = useful_stuff["train_auc"][-1]

    print("\n[Training]\n")
    print("total sample:", (TP + FP + TN + FN))
    print("TP: {:.2f}".format(TP))
    print("FP: {:.2f}".format(FP))
    print("TN: {:.2f}".format(TN))
    print("FN: {:.2f}".format(FN))
    print("acc: {:.2f}".format(acc))
    print("loss: {:.2f}".format(loss))
    print("recall: {:.2f}".format(recall))
    print("specificity: {:.2f}".format(specificity))
    print("precision: {:.2f}".format(precision))
    print("npv: {:.2f}".format(npv))
    print("f1: {:.2f}".format(f1))
    print("mcc: {:.2f}".format(mcc))
    print("auc: {:.2f}".format(auc))

    # save result
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["[Training average]"])
        writer.writerow(["ACC: {:.2}".format(acc)])
        writer.writerow(["LOSS: {:.2}".format(loss)])
        writer.writerow(["Recall: {:.2}".format(recall)])
        writer.writerow(["Specificity: {:.2}".format(specificity)])
        writer.writerow(["Precision: {:.2}".format(precision)])
        writer.writerow(["NPV: {:.2}".format(npv)])
        writer.writerow(["F1: {:.2}".format(f1)])
        writer.writerow(["MCC: {:.2}".format(mcc)])
        writer.writerow(["AUC: {:.2}".format(auc)])


def plot_lc_final(useful_stuff, fig_path, data_index):
    """
    plot learning curve.
    args:
        useful_stuff(dict): infomation dict, obtaining acc, loss, auc, ...
        fig_path(str): path to store figures
        data_index(int): what dataset is
    """
    # acc
    plt.plot(useful_stuff["train_acc"], label="train")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.title("train acc vs epochs")
    plt.grid()
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    fname = fig_path + "set-" + data_index + "-acc" + ".png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # loss
    plt.plot(useful_stuff["train_loss"], label="train")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train loss vs epochs")
    plt.grid()
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    fname = fig_path + "set-" + data_index + "-loss" + ".png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def read_data(pos_path, neg_path):
    """
    Open csv files as DataFrame
    """
    # positive
    df_pos = pd.read_csv(pos_path, encoding="utf-8", dtype=str).fillna("")
    df_pos = df_pos[["abstract"]]
    df_pos = df_pos.assign(label=1)

    # negative
    df_neg = pd.read_csv(neg_path, encoding="utf-8", dtype=str).fillna("")
    df_neg = df_neg[["abstract"]]
    df_neg = df_neg.assign(label=0)

    # combine pos + neg
    alldata = pd.concat([df_pos, df_neg], ignore_index=True)

    return alldata


def op_shuffle_data_eval(random_state):
    """
    Open data and return it as a DataFrame.
    This function only open neg_train-5 (for prediction).
    args:
        random_state(int): set random seed for shuffle
    """
    # set path
    model_index = str(input("Input model_index(0~4): "))
    model_path = "./model/balance/final/neg_train-" + model_index + ".pt"
    pickle_path = "./pickles/balance/1_dim_softmax-" + model_index
    pos_path = "./data/train.csv"
    neg_path = "./data/balance/neg_train/neg_train-5.csv"
    print()
    print("pos_path:", pos_path)
    print("neg_path:", neg_path)
    print("model_path:", model_path)
    print("pickle_path:", pickle_path)
    print()

    alldata = read_data(pos_path, neg_path)
    alldata = shuffle(alldata, random_state=random_state)

    return alldata, model_path, pickle_path


def op_shuffle_data_eval_exp(random_state):
    """
    Open data and return it as a DataFrame.
    This function only open neg_train-1 (for prediction).
    args:
        random_state(int): set random seed for shuffle
    """
    # set path
    model_index = str(input("Choose model_index(1~5) for evaluate data: "))
    model_path = "./model/balance/final/neg_train-" + model_index + ".pt"
    pickle_path = "./pickles/balance/exp-model-" + model_index  # save feature
    pos_path = "./data/train.csv"
    neg_path = "./data/balance/neg_train/neg_train-0.csv"
    print()
    print("pos_path:", pos_path)
    print("neg_path:", neg_path)
    print("model_path:", model_path)
    print("pickle_path:", pickle_path)
    print()

    alldata = read_data(pos_path, neg_path)
    alldata = shuffle(alldata, random_state=random_state)

    return alldata, model_path, pickle_path


def k_fold_tensor(X_train, y_train, k, random_state):
    """
    K-Fold Cross Validation.

    Input:
        X_train(tensor): prediction of abstract
        y_train(tensor): ground truth
    Output:
        train_list_x(list): list of list of abstract
        train_list_y(list): list of list of label
        cv_list_x(list): list of list of abstract
        cv_list_y(list): list of list of label
    """
    skf = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=random_state
    )  # init class

    train_list_x = []
    train_list_y = []
    cv_list_x = []
    cv_list_y = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        # train_index
        train_list_x.append(X_train[train_index])
        train_list_y.append(y_train[train_index])

        # test_index
        cv_list_x.append(X_train[test_index])
        cv_list_y.append(y_train[test_index])

    return train_list_x, train_list_y, cv_list_x, cv_list_y


def calc_metirc(yhat, y, TP, FP, TN, FN):
    # calu metric
    TP += ((yhat == 1) & (y == 1)).sum().item()
    FP += ((yhat == 1) & (y == 0)).sum().item()
    TN += ((yhat == 0) & (y == 0)).sum().item()
    FN += ((yhat == 0) & (y == 1)).sum().item()
    return TP, FP, TN, FN


def calc_auc(z, y, auc_y, auc_yhat):
    prob_2dim = F.softmax(z, dim=1)
    prob_1dim = prob_2dim[:, 1]
    auc_y.extend(y.cpu().detach().numpy())
    auc_yhat.extend(prob_1dim.cpu().detach().numpy())
    return auc_y, auc_yhat


def save_result(auc_y, auc_yhat, acc_, loss_, TP, FP, TN, FN, useful_stuff, mtype):

    fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
    auc = metrics.auc(fpr, tpr)

    useful_stuff[mtype + "_loss"].append(loss_)
    useful_stuff[mtype + "_acc"].append(acc_)
    useful_stuff[mtype + "_metric"].append((TP, FP, TN, FN))
    useful_stuff[mtype + "_auc"].append(auc)
    useful_stuff[mtype + "_fpr"].append(fpr)
    useful_stuff[mtype + "_tpr"].append(tpr)

    return useful_stuff


def train_ensemble(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    N_train,
    N_test,
    device,
    criterion,
    scheduler,
    epochs=20,
):

    useful_stuff = {
        "training_loss": [],
        "training_acc": [],
        "training_auc": [],
        "training_metric": [],
        "training_fpr": [],
        "training_tpr": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_metric": [],
        "train_fpr": [],
        "train_tpr": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_auc": [],
        "valid_metric": [],
        "valid_fpr": [],
        "valid_tpr": [],
    }

    for epoch in range(epochs):
        # training===========================================
        model.train()
        correct = 0
        training_loss = []
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y).sum().item()
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.data.item())

            # calu metric
            TP, FP, TN, FN = calc_metirc(yhat, y, TP, FP, TN, FN)

            # calu auc
            auc_y, auc_yhat = calc_auc(z, y, auc_y, auc_yhat)

        acc_ = correct / N_train
        loss_ = np.mean(training_loss)
        useful_stuff = save_result(
            auc_y, auc_yhat, acc_, loss_, TP, FP, TN, FN, useful_stuff, mtype="training"
        )

        # print("training loss: {0:.2f}".format(loss_))
        # print("training acc: {0:.2f}".format(acc_))
        # print('-' * 10)

        # train===========================================
        model.eval()
        correct = 0
        training_loss = []
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []
        with torch.no_grad():

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                z = model(x)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y).sum().item()
                loss = criterion(z, y)
                training_loss.append(loss.data.item())

                # calu metric
                TP, FP, TN, FN = calc_metirc(yhat, y, TP, FP, TN, FN)

                # calu auc
                auc_y, auc_yhat = calc_auc(z, y, auc_y, auc_yhat)

            loss_ = np.mean(training_loss)
            acc_ = correct / N_train

            useful_stuff = save_result(
                auc_y,
                auc_yhat,
                acc_,
                loss_,
                TP,
                FP,
                TN,
                FN,
                useful_stuff,
                mtype="train",
            )

            # print("train loss: {0:.2f}".format(loss_))
            # print("train acc: {0:.2f}".format(acc_))
            # print('-' * 10)

        # valid==================================================
        model.eval()
        correct = 0
        training_loss = []
        TP = FP = TN = FN = 0
        auc_y = []
        auc_yhat = []
        with torch.no_grad():
            for x, y in validation_dataloader:
                x, y = x.to(device), y.to(device)
                z = model(x)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y).sum().item()
                loss = criterion(z, y)
                training_loss.append(loss.data.item())

                # calu metric
                TP, FP, TN, FN = calc_metirc(yhat, y, TP, FP, TN, FN)

                # calu auc
                auc_y, auc_yhat = calc_auc(z, y, auc_y, auc_yhat)

            loss_ = np.mean(training_loss)
            acc_ = correct / N_test
            useful_stuff = save_result(
                auc_y,
                auc_yhat,
                acc_,
                loss_,
                TP,
                FP,
                TN,
                FN,
                useful_stuff,
                mtype="valid",
            )

            # print("valid loss: {0:.2f}".format(loss_))
            # print("valid acc: {0:.2f}".format(acc_))
            # print('-' * 10)

        # learning rate scheduler <---------------------------------------------------------
        scheduler.step(loss_)

    return useful_stuff


def eval_data_wiohout_train(
    model,
    train_dataloader,
    validation_dataloader,
    N_train,
    N_test,
    device,
    criterion,
):
    """
    eval the data.
    """

    useful_stuff = {
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_metric": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_auc": [],
        "valid_metric": [],
    }

    # train data
    model.eval()
    correct = 0
    training_loss = []
    TP = FP = TN = FN = 0
    auc_y = []
    auc_yhat = []

    with torch.no_grad():
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y).sum().item()
            loss = criterion(z, y)
            training_loss.append(loss.data.item())

            # calu metric
            TP += ((yhat == 1) & (y == 1)).sum().item()
            FP += ((yhat == 1) & (y == 0)).sum().item()
            TN += ((yhat == 0) & (y == 0)).sum().item()
            FN += ((yhat == 0) & (y == 1)).sum().item()

            # calu auc
            auc_y.extend(y.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        loss_ = np.mean(training_loss)
        acc_ = correct / N_train

        useful_stuff["train_loss"].append(loss_)
        useful_stuff["train_acc"].append(acc_)
        useful_stuff["train_metric"].append((TP, FP, TN, FN))
        useful_stuff["train_auc"].append(auc)

    # validation data
    model.eval()
    correct = 0
    training_loss = []
    TP = FP = TN = FN = 0
    auc_y = []
    auc_yhat = []

    with torch.no_grad():
        for x, y in validation_dataloader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y).sum().item()
            loss = criterion(z, y)
            training_loss.append(loss.data.item())

            # calu metric
            TP += ((yhat == 1) & (y == 1)).sum().item()
            FP += ((yhat == 1) & (y == 0)).sum().item()
            TN += ((yhat == 0) & (y == 0)).sum().item()
            FN += ((yhat == 0) & (y == 1)).sum().item()

            # calu auc
            auc_y.extend(y.cpu().detach().numpy())
            auc_yhat.extend(yhat.cpu().detach().numpy())

        fpr, tpr, _ = metrics.roc_curve(auc_y, auc_yhat)
        auc = metrics.auc(fpr, tpr)
        loss_ = np.mean(training_loss)
        acc_ = correct / N_test

        useful_stuff["valid_loss"].append(loss_)
        useful_stuff["valid_acc"].append(acc_)
        useful_stuff["valid_metric"].append((TP, FP, TN, FN))
        useful_stuff["valid_auc"].append(auc)

    return useful_stuff


def save_metrics(
    path,
    train_metric,
    ACC,
    LOSS,
    RECALL,
    SPECIFICITY,
    PRECISION,
    NPV,
    F1,
    MCC,
    AUC,
):
    """
    save metrics as csv files
    """
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        if train_metric:
            writer.writerow(["[Train average]"])
        else:
            writer.writerow(["[valid average]"])
        writer.writerow(["ACC: {:.2}".format((np.mean(ACC)))])
        writer.writerow(["LOSS: {:.2}".format(np.mean(LOSS))])
        writer.writerow(["Recall: {:.2}".format(np.mean(RECALL))])
        writer.writerow(["Specificity: {:.2}".format(np.mean(SPECIFICITY))])
        writer.writerow(["Precision: {:.2}".format(np.mean(PRECISION))])
        writer.writerow(["NPV: {:.2}".format(np.mean(NPV))])
        writer.writerow(["F1: {:.2}".format(np.mean(F1))])
        writer.writerow(["MCC: {:.2}".format(np.mean(MCC))])
        writer.writerow(["AUC: {:.2}".format(np.mean(AUC))])


def calc_metric_ensemble(training_history, fig_path, train_metric=True, detail=False):
    """
    Calculate metric.
    """
    # init
    ACC = []
    LOSS = []
    RECALL = []
    SPECIFICITY = []
    PRECISION = []
    NPV = []
    F1 = []
    MCC = []
    AUC = []
    FPR = []
    TPR = []

    for i in range(len(training_history)):

        if train_metric:
            (TP, FP, TN, FN) = training_history[i]["train_metric"][-1]
            auc = training_history[i]["train_auc"][-1]
            fpr = training_history[i]["train_fpr"][-1]
            tpr = training_history[i]["train_tpr"][-1]
            loss = training_history[i]["train_loss"][-1]
            path = fig_path + "train.txt"
        else:
            (TP, FP, TN, FN) = training_history[i]["valid_metric"][-1]
            auc = training_history[i]["valid_auc"][-1]
            fpr = training_history[i]["valid_fpr"][-1]
            tpr = training_history[i]["valid_tpr"][-1]
            loss = training_history[i]["valid_loss"][-1]
            path = fig_path + "valid.txt"

        acc = (TP + TN) / (TP + FP + TN + FN)

        try:
            recall = TP / (TP + FN)  # 召回率是在所有正樣本當中，能夠預測多少正樣本的比例
        except:
            recall = 0

        try:
            specificity = TN / (TN + FP)  # 特異度是在所有負樣本當中，能夠預測多少負樣本的比例
        except:
            specificity = 0

        try:
            precision = TP / (TP + FP)  # 準確率為在所有預測為正樣本中，有多少為正樣本
        except:
            precision = 0

        try:
            npv = TN / (TN + FN)  # npv為在所有預測為正樣本中，有多少為正樣本
        except:
            npv = 0

        try:
            f1 = (2 * recall * precision) / (recall + precision)  # F1-score則是兩者的調和平均數
        except:
            f1 = 0

        try:
            mcc = (TP * TN - FP * FN) / np.sqrt(
                ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            )
        except:
            mcc = 0

        if detail:
            print("[ fold", i, "]", "(Total sample: {})".format(TP + FP + TN + FN))
            print("TP:", TP)
            print("FP:", FP)
            print("TN:", TN)
            print("FN:", FN)
            print()
            print("acc:", acc)
            print("loss:", loss)
            print("recall:", recall)
            print("specificity:", specificity)
            print("precision:", precision)
            print("npv:", npv)
            print("f1:", f1)
            print("mcc:", mcc)
            print("auc:", auc)
            print("fpr:", fpr)
            print("tpr:", tpr)
            print("=" * 40)

        ACC.append(acc)
        LOSS.append(loss)
        RECALL.append(recall)
        SPECIFICITY.append(specificity)
        PRECISION.append(precision)
        NPV.append(npv)
        F1.append(f1)
        MCC.append(mcc)
        AUC.append(auc)
        FPR.append(fpr)
        TPR.append(tpr)

    if train_metric:
        print("\n[Training average]\n")
    else:
        print("\n[valid average]\n")
    print("ACC: {:.2}".format((np.mean(ACC))))
    print("LOSS: {:.2}".format(np.mean(LOSS)))
    print()
    print("Recall: {:.2}".format(np.mean(RECALL)))
    print("Specificity: {:.2}".format(np.mean(SPECIFICITY)))
    print("Precision: {:.2}".format(np.mean(PRECISION)))
    print("NPV: {:.2}".format(np.mean(NPV)))
    print()
    print("F1: {:.2}".format(np.mean(F1)))
    print("MCC: {:.2}".format(np.mean(MCC)))
    print("AUC: {:.2}".format(np.mean(AUC)))
    print()

    # save result
    save_metrics(
        path,
        train_metric,
        ACC,
        LOSS,
        RECALL,
        SPECIFICITY,
        PRECISION,
        NPV,
        F1,
        MCC,
        AUC,
    )


def plot_lc_ensemble(training_history, fig_path):

    a1, a2, a3, a4 = calc_avg(training_history)

    # color
    tr_color = ["#2ff5f2", "#2ff5e8", "#2ff5c0", "#2fbdf5", "#2f99f5"]
    val_color = ["#f5952f", "#f5ac2f", "#f5c02f", "#f5d72f", "#f5ee2f"]

    # train loss
    for idx, color in enumerate(tr_color):
        plt.plot(
            training_history[idx]["train_loss"],
            "--",
            alpha=0.4,
            label="train" + str(idx),
            color=color,
        )
    plt.plot(a1, label="average training")

    # valid loss
    for idx, color in enumerate(val_color):
        plt.plot(
            training_history[idx]["valid_loss"],
            "--",
            alpha=0.4,
            label="valid" + str(idx),
            color=color,
        )

    plt.plot(a2, label="average valid")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.title("training / valid loss vs iterations")
    plt.grid()
    plt.savefig(fig_path + "loss" + ".png", bbox_inches="tight")
    plt.close()

    # train acc
    for idx, color in enumerate(tr_color):
        plt.plot(
            training_history[idx]["train_acc"],
            "--",
            alpha=0.4,
            label="train" + str(idx),
            color=color,
        )
    plt.plot(a3, label="average training")

    # valid acc
    for idx, color in enumerate(val_color):
        plt.plot(
            training_history[idx]["valid_acc"],
            "--",
            alpha=0.4,
            label="valid" + str(idx),
            color=color,
        )
    plt.plot(a4, label="average valid")
    plt.ylabel("acc")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    plt.legend()
    plt.title("training / valid acc vs iterations")
    plt.grid()
    plt.savefig(fig_path + "acc" + ".png", bbox_inches="tight")
    plt.close()

    # roc
    plot_roc(training_history, fig_path, mtype="train")
    plot_roc(training_history, fig_path, mtype="valid")


def plot_roc(training_history, fig_path, mtype="train"):
    """
    plot roc curve and save as png
    """
    for i in range(len(training_history)):
        auc = training_history[i][mtype + "_auc"][-1]
        fpr = training_history[i][mtype + "_fpr"][-1]
        tpr = training_history[i][mtype + "_tpr"][-1]

        plt.plot(fpr, tpr, label="Fold-" + str(i) + " AUC = %0.2f" % auc)

    plt.title(mtype + " roc curve")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(fig_path + mtype + "-roc" + ".png", bbox_inches="tight")
    plt.close()


def concate_feature(path_list, pathy):
    """
    concate feature x.
    args:
        path_list(list): list containing feature path
        pathy(str): target path
    """

    X_list = []
    for path in path_list:
        with open(path, "rb") as f:
            print("load path:", path)
            X_list.append(torch.tensor(pickle.load(f)).reshape(-1, 1))  # check_dim

    # load target
    with open(pathy, "rb") as f:
        print("load path:", pathy)
        y_list = pickle.load(f)

    # stacking 5 features (1280, 1) -> (1280, 5) / (1280, 10)
    X = torch.cat((X_list[0], X_list[1], X_list[2], X_list[3], X_list[4]), 1)
    print("X.shape:", X.shape)
    # label
    y = torch.tensor(y_list)
    print("y.shape:", y.shape)
    return X, y


def op_data_ensemble(exp=False, shuffle=False):
    """
    Open data and return as tensor.

    input:
        exp(bool): whether in experiment
        shuffle(bool): whether shuffle data

    output:
        X(tensor): Features with shape (len, 5)
        y(tesnor): Leabel with shape (len)
    """

    # set path
    pathx_1 = "./pickles/balance/1_dim_softmax-0-X.pkl"
    pathx_2 = "./pickles/balance/1_dim_softmax-1-X.pkl"
    pathx_3 = "./pickles/balance/1_dim_softmax-2-X.pkl"
    pathx_4 = "./pickles/balance/1_dim_softmax-3-X.pkl"
    pathx_5 = "./pickles/balance/1_dim_softmax-4-X.pkl"
    pathy = "./pickles/balance/1_dim_softmax-y.pkl"  # all the same

    if not exp:
        fig_path = "./pics/balance/ensemble/"
        model_path = "./model/balance/ensemble/"
        history_path = "./history/balance/ensemble/ensemble.pkl"
    else:
        fig_path = "./pics/balance/ensemble/exp/"
        model_path = "./model/balance/ensemble/exp/"
        history_path = "./history/balance/ensemble/exp/ensemble.pkl"

    print("fig_path:", fig_path)
    print("model_path:", model_path)
    print("history_path:", history_path)
    print("shuffle data:", shuffle)
    print()

    # load feture
    path_list = [pathx_1, pathx_2, pathx_3, pathx_4, pathx_5]
    X, y = concate_feature(path_list, pathy)

    # shuffle data
    if shuffle:
        rand_idx = torch.randperm(X.size()[0])
        X = X[rand_idx]
        y = y[rand_idx]

    return X, y, fig_path, model_path, history_path


def op_data_ensemble_exp(exp=False, shuffle=False):
    """
    Open data and return as tensor.

    input:
        exp(bool): whether in experiment
        shuffle(bool): whether shuffle data

    output:
        X(tensor): Features with shape (len, 5)
        y(tesnor): Leabel with shape (len)
    """
    # set path
    pathx_1 = "./pickles/balance/exp-model-1-X.pkl"
    pathx_2 = "./pickles/balance/exp-model-2-X.pkl"
    pathx_3 = "./pickles/balance/exp-model-3-X.pkl"
    pathx_4 = "./pickles/balance/exp-model-4-X.pkl"
    pathx_5 = "./pickles/balance/exp-model-5-X.pkl"
    pathy = "./pickles/balance/exp-model-1-y.pkl"  # all the same

    if not exp:
        fig_path = "./pics/balance/ensemble/"
        model_path = "./model/balance/ensemble/"
        history_path = "./history/balance/ensemble/ensemble.pkl"
    else:
        fig_path = "./pics/balance/ensemble/exp/"
        model_path = "./model/balance/ensemble/exp/"
        history_path = "./history/balance/ensemble/exp/ensemble.pkl"

    print("fig_path:", fig_path)
    print("model_path:", model_path)
    print("history_path:", history_path)
    print("shuffle data:", shuffle)
    print()

    # load feture
    path_list = [pathx_1, pathx_2, pathx_3, pathx_4, pathx_5]
    X_list = []
    for path in path_list:
        with open(path, "rb") as f:
            print("load feature:", path)
            X_list.append(torch.tensor(pickle.load(f)).reshape(-1, 1))

    # load target
    with open(pathy, "rb") as f:
        print("load target:", pathy)
        y_list = pickle.load(f)

    # stacking 5 features (1280, 1) -> (1280, 5)
    X = torch.cat((X_list[0], X_list[1], X_list[2], X_list[3], X_list[4]), 1)

    # label
    y = torch.tensor(y_list)

    # shuffle data
    if shuffle:
        rand_idx = torch.randperm(X.size()[0])
        X = X[rand_idx]
        y = y[rand_idx]

    return X, y, fig_path, model_path, history_path


class Data(Dataset):
    # Constructor
    def __init__(self, X, y):
        self.x = X.float()
        self.y = y.long()
        self.len = self.y.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


class Net(nn.Module):
    # Constructor
    def __init__(self, input_dim, neuron=10, p=0):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.neuron = neuron
        self.layer = torch.nn.Sequential(
            nn.Dropout(p=0.1),
            # 1
            nn.Linear(self.input_dim, self.neuron),
            nn.BatchNorm1d(self.neuron),
            nn.ReLU(),
            nn.Dropout(p=p),
            # 2
            nn.Linear(self.neuron, self.neuron),
            nn.BatchNorm1d(self.neuron),
            nn.ReLU(),
            nn.Dropout(p=p),
            # 3
            nn.Linear(self.neuron, 2),
            # nn.Linear(5, 2)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Linear_net(nn.Module):
    # Constructor
    def __init__(self, input_dim):
        super(Linear_net, self).__init__()
        self.input_dim = input_dim
        self.layer = torch.nn.Sequential(nn.Linear(self.input_dim, 2))

    def forward(self, x):
        x = self.layer(x)
        return x
