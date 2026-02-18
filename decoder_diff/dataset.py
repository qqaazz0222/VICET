import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DiffDataset(Dataset):
    def __init__(self, feature_dir, diff_dir, transform=None):
        self.feature_dir = feature_dir
        self.diff_dir = diff_dir
        self.transform = transform
        
        # Get list of feature files
        self.feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
        
        # Filter files that have corresponding diff files
        self.valid_files = []
        for f_path in self.feature_files:
            basename = os.path.basename(f_path)
            diff_path = os.path.join(diff_dir, basename)
            if os.path.exists(diff_path):
                self.valid_files.append(basename)
            else:
                print(f"Warning: Diff file not found for {basename}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        basename = self.valid_files[idx]
        
        # Load feature
        # Shape: (2304, 1024) -> reshape to (1024, 48, 48)
        feature_path = os.path.join(self.feature_dir, basename)
        feature = np.load(feature_path)
        feature = feature.reshape(48, 48, 1024).transpose(2, 0, 1) # (C, H, W)
        feature = torch.from_numpy(feature).float()
        
        # Load diff
        # Shape: (512, 512)
        diff_path = os.path.join(self.diff_dir, basename)
        diff = np.load(diff_path)
        diff = torch.from_numpy(diff).float().unsqueeze(0) # (1, H, W)
        
        if self.transform:
            # Apply transforms if needed
            pass
            
        return feature, diff
