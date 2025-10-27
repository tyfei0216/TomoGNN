import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import utils


class DataForUnet(Dataset):
    def __init__(self, df, filter_class=1, augment=True):
        """
        Dataset for 2D slices from 3D tomograms.

        Args:
            df: DataFrame with columns 'data' (path to tomogram) and 'label_path' (path to label)
            filter_class: class ID to filter labels (default: 1)
            augment: whether to apply data augmentation (default: True)
        """
        self.augment = augment
        self.slices = []
        self.labels = []

        for i, r in df.iterrows():
            # Load tomogram and label
            d = utils.readTomogram(r["data"])  # shape: (D, H, W)
            label = utils.readTomogram(r["labels"])  # shape: (D, H, W)

            # Process binary label
            binary_label = (label == filter_class).astype(np.float32)

            # Filter out empty slices (slices with no labels)
            for z in range(d.shape[0]):
                slice_data = d[z]  # (H, W)
                slice_label = binary_label[z]  # (H, W)

                # Skip empty slices (no positive labels)
                if slice_label.sum() == 0:
                    continue

                # Normalize to mean 0, std 1
                slice_data = slice_data.astype(np.float32)
                slice_mean = slice_data.mean()
                slice_std = slice_data.std()
                if slice_std > 1e-6:
                    slice_data = (slice_data - slice_mean) / slice_std
                else:
                    slice_data = slice_data - slice_mean

                self.slices.append(slice_data)
                self.labels.append(slice_label)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_data = self.slices[idx].copy()  # (1024, 1024)
        slice_label = self.labels[idx].copy()  # (1024, 1024)

        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                slice_data = np.flip(slice_data, axis=0).copy()
                slice_label = np.flip(slice_label, axis=0).copy()

            # Random vertical flip
            if np.random.rand() < 0.5:
                slice_data = np.flip(slice_data, axis=1).copy()
                slice_label = np.flip(slice_label, axis=1).copy()

            # Random 90-degree rotation
            k = np.random.randint(0, 4)  # 0, 1, 2, or 3 rotations of 90 degrees
            if k > 0:
                slice_data = np.rot90(slice_data, k=k).copy()
                slice_label = np.rot90(slice_label, k=k).copy()

        # Resize to 512x512
        slice_data = cv2.resize(slice_data, (512, 512), interpolation=cv2.INTER_LINEAR)
        slice_label = cv2.resize(
            slice_label, (512, 512), interpolation=cv2.INTER_NEAREST
        )

        # Add channel dimension: (512, 512) -> (1, 512, 512)
        slice_data = np.expand_dims(slice_data, axis=0)
        slice_label = np.expand_dims(slice_label, axis=0)

        # Convert to torch tensors
        slice_data = torch.from_numpy(slice_data).float()
        slice_label = torch.from_numpy(slice_label).float()

        return slice_data, slice_label
