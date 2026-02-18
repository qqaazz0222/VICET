import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LocationDataset(Dataset):
    def __init__(self, feature_dir, location_dir, transform=None):
        self.feature_dir = feature_dir
        self.location_dir = location_dir
        self.transform = transform
        
        # Get list of feature files
        self.feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
        
        # Filter files that have corresponding location files
        self.valid_files = []
        for f_path in self.feature_files:
            basename = os.path.basename(f_path)
            loc_path = os.path.join(location_dir, basename)
            if os.path.exists(loc_path):
                self.valid_files.append(basename)
            else:
                print(f"Warning: Location file not found for {basename}")

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
        
        # Load location
        # Shape: (1, 16, 16)
        loc_path = os.path.join(self.location_dir, basename)
        location = np.load(loc_path)
        
        # Ensure shape (1, 512, 512)
        if location.ndim == 2:
            location = location[np.newaxis, ...]
        elif location.ndim == 3 and location.shape[2] == 1:
             location = location.transpose(2, 0, 1)
        
        location = torch.from_numpy(location).float()
        
        if self.transform:
            pass
            
        return feature, location
