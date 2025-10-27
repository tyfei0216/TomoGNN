import numpy as np
import torch
from torch.utils.data import Dataset

import utils


class Particle3DDataset(Dataset):
    """
    Dataset for extracting 3D subvolumes centered at particle coordinates for binary classification.
    Each sample is a (1, 65, 65, 65) volume and a label (0/1).

    Args:
        df (pd.DataFrame): DataFrame with columns ['z', 'y', 'x', 'label', 'tomogram']
            - 'z', 'y', 'x': center coordinates (int)
            - 'label': 0 (neg) or 1 (pos)
            - 'tomogram': path to tomogram file (all rows can share the same path, or be per-row)
        volume_cache (dict, optional): If provided, maps tomogram path to loaded 3D numpy array.
        crop_size (int): Size of the cubic patch to extract (default 65).
        norm (str): Normalization method ('zscore', 'hist', or None).
    """

    def __init__(self, df, tomo_paths, crop_size=65, norm=None, r=0):
        self.df = df.reset_index(drop=True)
        self.crop_size = crop_size
        self.norm = norm
        self.r = r
        # Collect all unique tomogram paths
        # tomo_paths = self.df["tomogram"].unique()
        self.loaded_volumes = {}
        # cache = volume_cache if volume_cache is not None else {}
        for path in tomo_paths:
            # if path in cache:
            #     vol = cache[path]
            # else:
            vol = utils.readTomogram(path)
            # Normalize each 2D slice
            if self.norm == "zscore":
                vol = np.stack(
                    [(sl - sl.mean()) / (sl.std() + 1e-6) for sl in vol], axis=0
                )
            elif self.norm == "hist":
                from skimage import exposure

                vol = np.stack([exposure.equalize_hist(sl) for sl in vol], axis=0)
            self.loaded_volumes[path] = vol

    def __len__(self):
        return len(self.df)

    def _get_volume(self, path):
        return self.loaded_volumes[path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        label = int(row["label"])
        tomo_path = row["tomogram"]
        vol = self._get_volume(tomo_path)
        D, H, W = vol.shape
        sz = self.crop_size // 2
        # Random center shift
        if self.r > 0:
            dz = np.random.uniform(-self.r, self.r)
            dy = np.random.uniform(-self.r, self.r)
            dx = np.random.uniform(-self.r, self.r)
            z = int(np.clip(z + dz, 0, D - 1))
            y = int(np.clip(y + dy, 0, H - 1))
            x = int(np.clip(x + dx, 0, W - 1))
        # Clamp to valid range
        z1, z2 = max(z - sz, 0), min(z + sz + 1, D)
        y1, y2 = max(y - sz, 0), min(y + sz + 1, H)
        x1, x2 = max(x - sz, 0), min(x + sz + 1, W)
        crop = np.zeros(
            (self.crop_size, self.crop_size, self.crop_size), dtype=vol.dtype
        )
        cz1, cy1, cx1 = sz - (z - z1), sz - (y - y1), sz - (x - x1)
        cz2, cy2, cx2 = cz1 + (z2 - z1), cy1 + (y2 - y1), cx1 + (x2 - x1)
        crop[cz1:cz2, cy1:cy2, cx1:cx2] = vol[z1:z2, y1:y2, x1:x2]
        # Data augmentation: random flip and 90-degree rotation
        # Flip along axes
        for axis in range(3):
            if np.random.rand() < 0.5:
                crop = np.flip(crop, axis=axis)
        # Random 90-degree rotation about z axis
        k = np.random.randint(0, 4)
        crop = np.rot90(crop, k=k, axes=(1, 2))
        crop = crop.astype(np.float32)
        crop = np.expand_dims(crop, 0)  # (1, D, H, W)
        return torch.from_numpy(crop), torch.tensor(label, dtype=torch.float32)
