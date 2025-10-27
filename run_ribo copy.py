import sys

sys.path.append("/home/feity/cryoem/")
import importlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import data
import modules
import postprocess
import utils

all_tomos = [
    "24092002",
    "24092013",
    "24092016",
    "24092020",
    "24092024",
    "24092026",
    "24092030",
    "24092009",
    "24092014",
    "24092017",
    "24092023",
    "24092025",
    "24092029",
    "24092031",
]

all_res_auroc = {}
all_res_acc = {}
all_res_aupr = {}
all_res_f1 = {}

for name in all_tomos:
    print(f"\n{'='*60}")
    print(f"Processing tomogram: {name}")
    print(f"{'='*60}")
    ribo_adata = sc.read_h5ad(f"/home/feity/cryoem/temp/ribo_results/%s.h5ad" % name)

    X = ribo_adata.X.astype(np.float32)
    # Y: 4 possible classes (0,1,2,3)
    # Y = ribo_adata.obs["leiden"].astype(int).values
    # Y  = ribo_adata.obs["class_num"].astype(int).values
    # Y  = ribo_adata.obs["label"].astype(int).values
    Y = ribo_adata.obs["class_num2"].astype(int).values
    num_classes = 2
    X = X[Y != 0]
    Y = Y[Y != 0] - 1
    a, b = np.unique(Y, return_counts=True)
    weights = np.zeros(num_classes)
    for i, c in zip(a, b):
        weights[i] = np.sum(b) / c

    # Store results for all 5 folds
    fold_auroc = []
    fold_acc = []
    fold_aupr = []
    fold_f1 = []

    # ===== 5-Fold Cross Validation =====
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        print(f"\nFold {fold_idx}/5")
        print("-" * 40)

        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        # ===== Simple Fully Connected Network =====
        class SimpleFC(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(SimpleFC, self).__init__()

                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, num_classes),
                )

            def forward(self, x):
                out = self.net(x)
                return out

        # Instantiate model
        model = SimpleFC(input_size=262, hidden_size=64, num_classes=num_classes)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # ===== Training loop =====
        max_acc = 0.0
        best_model_state = None
        epochs = 2000
        for epoch in range(epochs):
            model.train()
            outputs = model(X_train_t)
            # print(outputs.shape, y_train_t.shape)
            loss = criterion(outputs, y_train_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    preds = torch.argmax(model(X_test_t), dim=1)
                    acc = accuracy_score(y_test_t.numpy(), preds.numpy())
                print(
                    f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Acc: {acc:.4f}"
                )
                if acc > max_acc:
                    max_acc = acc
                    best_model_state = model.state_dict().copy()

        # ===== Final evaluation for this fold =====
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            preds = torch.argmax(model(X_test_t), dim=1)
            final_acc = accuracy_score(y_test_t.numpy(), preds.numpy())

        from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

        # Get probabilities for AUROC calculation
        with torch.no_grad():
            y_proba = torch.softmax(model(X_test_t), dim=1)[:, 1].numpy()

        try:
            auroc = roc_auc_score(y_test_t.numpy(), y_proba)
        except:
            auroc = np.nan

        precision, recall, _ = precision_recall_curve(y_test_t.numpy(), y_proba)
        aupr = auc(recall, precision)
        f1 = f1_score(y_test_t.numpy(), preds.numpy())

        print(f"  Fold {fold_idx} Results:")
        print(f"    Accuracy: {final_acc:.4f}")
        print(f"    AUROC: {auroc:.4f}")
        print(f"    AUPR: {aupr:.4f}")
        print(f"    F1-score: {f1:.4f}")

        # Store results for this fold
        fold_auroc.append(auroc)
        fold_acc.append(final_acc)
        fold_aupr.append(aupr)
        fold_f1.append(f1)

    # Store all fold results for this tomogram
    all_res_auroc[name] = fold_auroc
    all_res_acc[name] = fold_acc
    all_res_aupr[name] = fold_aupr
    all_res_f1[name] = fold_f1

    # Print summary for this tomogram
    print(f"\n{name} Summary (5-fold CV):")
    print(f"  Accuracy:  {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"  AUROC:     {np.nanmean(fold_auroc):.4f} ± {np.nanstd(fold_auroc):.4f}")
    print(f"  AUPR:      {np.mean(fold_aupr):.4f} ± {np.std(fold_aupr):.4f}")
    print(f"  F1-score:  {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

with open("/home/feity/cryoem/temp/ribo_results/res.pkl", "wb") as f:
    pickle.dump(
        {
            "roc": all_res_auroc,
            "acc": all_res_acc,
            "aupr": all_res_aupr,
            "f1": all_res_f1,
        },
        f,
    )
