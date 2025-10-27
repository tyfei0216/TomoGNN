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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import data
import modules
import postprocess
import utils

all_tomos = ["fold1", "fold2"]

all_res_mse = {}
all_res_mae = {}
all_predictions = []  # Store predictions for each particle

for name in all_tomos:
    print(f"\n{'='*60}")
    print(f"Processing tomogram: {name}")
    print(f"{'='*60}")
    ribo_adata = sc.read_h5ad(f"/home/feity/cryoem/temp/ribo_results/%s.h5ad" % name)

    X = ribo_adata.X.astype(np.float32)

    # Extract angle columns and convert to sin/cos (6 output values)
    angles_rot = np.deg2rad(ribo_adata.obs["rlnAngleRot"].values)
    angles_tilt = np.deg2rad(ribo_adata.obs["rlnAngleTilt"].values)
    angles_psi = np.deg2rad(ribo_adata.obs["rlnAnglePsi"].values)

    # Create targets: [sin_rot, cos_rot, sin_tilt, cos_tilt, sin_psi, cos_psi]
    Y = np.column_stack(
        [
            np.sin(angles_rot),
            np.cos(angles_rot),
            np.sin(angles_tilt),
            np.cos(angles_tilt),
            np.sin(angles_psi),
            np.cos(angles_psi),
        ]
    ).astype(np.float32)

    # Get the centers (x, y, z) from obs
    centers_x = ribo_adata.obs["x"].values
    centers_y = ribo_adata.obs["y"].values
    centers_z = ribo_adata.obs["z"].values

    # Filter out rows where class is unknown (if needed)
    if "class" in ribo_adata.obs.columns:
        mask = ribo_adata.obs["class"] != "unknown0"
        X = X[mask]
        Y = Y[mask]
        centers_x = centers_x[mask]
        centers_y = centers_y[mask]
        centers_z = centers_z[mask]

    # Store results for all 5 folds
    fold_mse = []
    fold_mae = []

    # ===== 5-Fold Cross Validation =====
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold_idx}/5")
        print("-" * 40)

        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # Split centers for this fold
        centers_x_test = centers_x[test_idx]
        centers_y_test = centers_y[test_idx]
        centers_z_test = centers_z[test_idx]

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        # ===== Simple Fully Connected Network for Regression =====
        class SimpleFC(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleFC, self).__init__()

                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                out = self.net(x)
                return out

        # Instantiate model for regression (6 output values)
        model = SimpleFC(input_size=262, hidden_size=128, output_size=6)
        # Loss and optimizer for regression
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # ===== Training loop =====
        best_loss = float("inf")
        best_model_state = None
        epochs = 2000
        for epoch in range(epochs):
            model.train()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    test_loss = criterion(test_outputs, y_test_t).item()
                print(
                    f"  Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}"
                )
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model_state = model.state_dict().copy()

        # ===== Final evaluation for this fold =====
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            preds = model(X_test_t).numpy()
            targets = y_test_t.numpy()

        # Calculate MSE and MAE
        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)

        print(f"  Fold {fold_idx} Results:")
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.6f}")

        # Store predictions for each particle
        for i in range(len(test_idx)):
            all_predictions.append(
                {
                    "tomogram": name,
                    "fold": fold_idx,
                    "x": centers_x_test[i],
                    "y": centers_y_test[i],
                    "z": centers_z_test[i],
                    "pred_sin_rot": preds[i, 0],
                    "pred_cos_rot": preds[i, 1],
                    "pred_sin_tilt": preds[i, 2],
                    "pred_cos_tilt": preds[i, 3],
                    "pred_sin_psi": preds[i, 4],
                    "pred_cos_psi": preds[i, 5],
                    "target_sin_rot": targets[i, 0],
                    "target_cos_rot": targets[i, 1],
                    "target_sin_tilt": targets[i, 2],
                    "target_cos_tilt": targets[i, 3],
                    "target_sin_psi": targets[i, 4],
                    "target_cos_psi": targets[i, 5],
                }
            )

        # Store results for this fold
        fold_mse.append(mse)
        fold_mae.append(mae)

    # Store all fold results for this tomogram
    all_res_mse[name] = fold_mse
    all_res_mae[name] = fold_mae

    # Print summary for this tomogram
    print(f"\n{name} Summary (5-fold CV):")
    print(f"  MSE:  {np.mean(fold_mse):.6f} ± {np.std(fold_mse):.6f}")
    print(f"  MAE:  {np.mean(fold_mae):.6f} ± {np.std(fold_mae):.6f}")

# Create results dataframe
results_data = []
for name in all_tomos:
    for fold_idx in range(5):
        results_data.append(
            {
                "tomogram": name,
                "fold": fold_idx + 1,
                "mse": all_res_mse[name][fold_idx],
                "mae": all_res_mae[name][fold_idx],
            }
        )

results_df = pd.DataFrame(results_data)

# Create predictions dataframe
predictions_df = pd.DataFrame(all_predictions)

# Save dataframes to CSV
results_df.to_csv(
    "/home/feity/cryoem/temp/ribo_results/angle_regression_results.csv", index=False
)
predictions_df.to_csv(
    "/home/feity/cryoem/temp/ribo_results/angle_predictions.csv", index=False
)

# Also save summary statistics
summary_data = []
for name in all_tomos:
    summary_data.append(
        {
            "tomogram": name,
            "mean_mse": np.mean(all_res_mse[name]),
            "std_mse": np.std(all_res_mse[name]),
            "mean_mae": np.mean(all_res_mae[name]),
            "std_mae": np.std(all_res_mae[name]),
        }
    )

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(
    "/home/feity/cryoem/temp/ribo_results/angle_regression_summary.csv", index=False
)

print("\n" + "=" * 60)
print("Results saved:")
print(
    "  - Per-fold results: /home/feity/cryoem/temp/ribo_results/angle_regression_results.csv"
)
print(
    "  - Summary statistics: /home/feity/cryoem/temp/ribo_results/angle_regression_summary.csv"
)
print(
    "  - Particle predictions: /home/feity/cryoem/temp/ribo_results/angle_predictions.csv"
)
print("=" * 60)
