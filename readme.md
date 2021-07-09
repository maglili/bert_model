# Bert model

Train different Bert model for crm abstract classification.

## Intro

**main workflow:**

1. `sub_model-k_fold.py`
2. `sub_model-final.py`
3. `eval_data_for_ensemble.py`
4. `ensemble-k_fold.py`

**helper scripts:**

1. `tools.py`
2. `test.py`
3. `eval_data_without_train-ensemble.py`

**structure:**

```text
./bert_model/
├── data
│   ├── balance
│   │   ├── neg_test
│   │   │   └── neg_test-0.csv
│   │   ├── neg_train
│   │   │   ├── neg_train-0.csv
│   │   │   ├── neg_train-1.csv
│   │   │   ├── neg_train-2.csv
│   │   │   ├── neg_train-3.csv
│   │   │   ├── neg_train-4.csv
│   │   │   └── neg_train-5.csv
│   │   └── neg_valid
│   │       └── neg_valid-0.csv
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── ensemble-k_fold.py
├── eval_data_for_ensemble.py
├── eval_data_without_train-ensemble.py
├── history
│   └── balance
│       ├── ensemble
│       │   ├── ensemble.pkl
│       │   └── exp
│       │       └── ensemble.pkl
│       ├── final
│       │   ├── neg_train-0.pkl
│       │   ├── neg_train-1.pkl
│       │   ├── neg_train-2.pkl
│       │   ├── neg_train-3.pkl
│       │   ├── neg_train-4.pkl
│       │   └── neg_train-5.pkl
│       └── kfold
│           ├── neg_train-0.pkl
│           ├── neg_train-1.pkl
│           ├── neg_train-2.pkl
│           ├── neg_train-3.pkl
│           └── neg_train-4.pkl
├── model
│   └── balance
│       ├── ensemble
│       ├── final
│       │   ├── neg_train-0.pt
│       │   ├── neg_train-1.pt
│       │   ├── neg_train-2.pt
│       │   ├── neg_train-3.pt
│       │   ├── neg_train-4.pt
│       │   └── neg_train-5.pt
│       └── kfold
│           ├── neg_train-0.pt
│           ├── neg_train-1.pt
│           ├── neg_train-2.pt
│           ├── neg_train-3.pt
│           └── neg_train-4.pt
├── pickles
│   └── balance
│       ├── 1_dim_softmax-0-X.pkl
│       ├── 1_dim_softmax-1-X.pkl
│       ├── 1_dim_softmax-2-X.pkl
│       ├── 1_dim_softmax-3-X.pkl
│       ├── 1_dim_softmax-4-X.pkl
│       └── 1_dim_softmax-y.pkl
├── pics
│   └── balance
│       ├── ensemble
│       │   ├── acc.png
│       │   ├── exp
│       │   │   ├── acc.png
│       │   │   ├── loss.png
│       │   │   ├── train-roc.png
│       │   │   ├── train.txt
│       │   │   ├── valid-roc.png
│       │   │   └── valid.txt
│       │   ├── loss.png
│       │   ├── train.txt
│       │   └── valid.txt
│       ├── final
│       │   ├── final-0.txt
│       │   ├── final-1.txt
│       │   ├── final-2.txt
│       │   ├── final-3.txt
│       │   ├── final-4.txt
│       │   ├── final-5.txt
│       │   ├── set-0-acc.png
│       │   ├── set-0-loss.png
│       │   ├── set-1-acc.png
│       │   ├── set-1-loss.png
│       │   ├── set-2-acc.png
│       │   ├── set-2-loss.png
│       │   ├── set-3-acc.png
│       │   ├── set-3-loss.png
│       │   ├── set-4-acc.png
│       │   ├── set-4-loss.png
│       │   ├── set-5-acc.png
│       │   └── set-5-loss.png
│       └── kfold
│           ├── set-0-acc.png
│           ├── set-0-loss.png
│           ├── set-1-acc.png
│           ├── set-1-loss.png
│           ├── set-2-acc.png
│           ├── set-2-loss.png
│           ├── set-3-acc.png
│           ├── set-3-loss.png
│           ├── set-4-acc.png
│           ├── set-4-loss.png
│           ├── train-0.txt
│           ├── train-1.txt
│           ├── train-2.txt
│           ├── train-3.txt
│           ├── train-4.txt
│           ├── valid-0.txt
│           ├── valid-1.txt
│           ├── valid-2.txt
│           ├── valid-3.txt
│           └── valid-4.txt
├── readme.md
├── sub_model-final.py
├── sub_model-k_fold.py
├── test.py
└── tools.py

24 directories, 99 files

```

