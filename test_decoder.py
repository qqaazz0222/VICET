import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import pydicom
from scipy.ndimage import gaussian_filter

def load_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load models dynamically
LocationModel = load_module_from_path('./decoder_location/model.py', 'location_model')
DiffModel = load_module_from_path('./decoder_diff/model.py', 'diff_model')

def test_decoders(input_path, loc_checkpoint, diff_checkpoint, output_dir, input_hu_array, gt_hu_array, gt_diff_file, gt_location_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Feature
    print(f"Loading input feature from {input_path}...")
    try:
        feature = np.load(input_path)
        # Reshape logic
        if feature.ndim == 2:
             # (N, 1024) -> (48, 48, 1024) -> (1024, 48, 48)
             feature = feature.reshape(48, 48, 1024).transpose(2, 0, 1)
        elif feature.ndim == 3 and feature.shape[0] == 1024:
             pass 
        elif feature.ndim == 3 and feature.shape[2] == 1024:
             feature = feature.transpose(2, 0, 1)
             
        feature_tensor = torch.from_numpy(feature).float().unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading feature: {e}")
        return

    # 2. Location Decoder Inference
    print(f"Running Location Decoder ({loc_checkpoint})...")
    try:
        loc_model = LocationModel.RefinedLocationDecoder().to(device)
        loc_model.load_state_dict(torch.load(loc_checkpoint, map_location=device, weights_only=True))
        loc_model.eval()
        
        with torch.no_grad():
            loc_out = loc_model(feature_tensor)
            loc_prob = torch.sigmoid(loc_out).cpu().numpy().squeeze()
            loc_binary = (loc_prob > 0.5).astype(float)
            
        # Save
        np.save(os.path.join(output_dir, 'location_prob.npy'), loc_prob)
        np.save(os.path.join(output_dir, 'location_binary.npy'), loc_binary)
        print("Location decoder output saved.")
        
    except Exception as e:
        print(f"Error in Location Decoder: {e}")
        loc_prob = None
        
    # 3. Diff Decoder Inference
    print(f"Running Diff Decoder ({diff_checkpoint})...")
    try:
        diff_model = DiffModel.RefinedDiffDecoder().to(device)
        diff_model.load_state_dict(torch.load(diff_checkpoint, map_location=device, weights_only=True))
        diff_model.eval()
        
        with torch.no_grad():
            diff_out = diff_model(feature_tensor)
            diff_res = diff_out.cpu().numpy().squeeze()
            
        # Save
        np.save(os.path.join(output_dir, 'diff_output.npy'), diff_res)
        print("Diff decoder output saved.")
        
    except Exception as e:
        print(f"Error in Diff Decoder: {e}")
        diff_res = None
        
    hu_loc_binary = np.zeros_like(loc_binary)
    hu_loc_binary[input_hu_array > 16] = 1
        
    ratio = 0.15
    final_res = diff_res * hu_loc_binary * ratio + diff_res * loc_binary * (1 - ratio)
    final_res = gaussian_filter(final_res, sigma=1.0)
    
    final_sythetic = input_hu_array + final_res
    
    # window center: 40, windwo width: 400
    window_center = 40
    window_width = 400
    input_hu_array = np.clip(input_hu_array, window_center - window_width / 2, window_center + window_width / 2)
    final_sythetic = np.clip(final_sythetic, window_center - window_width / 2, window_center + window_width / 2)
    final_gt = np.clip(gt_hu_array, window_center - window_width / 2, window_center + window_width / 2)
    gt_diff = np.load(gt_diff_file)
    gt_location = np.load(gt_location_file)

    # 4. Visualization
    plt.figure(figsize=(15, 10))
    
    if gt_location is not None:
        plt.subplot(3, 3, 1)
        plt.title("Location(GT)")
        plt.imshow(gt_location, cmap='gray')
        plt.colorbar()
    
    if gt_diff is not None:
        plt.subplot(3, 3, 2)
        plt.title("Diff(GT)")
        plt.imshow(gt_diff, cmap='gray')
        plt.colorbar()
    
    if loc_binary is not None:
        plt.subplot(3, 3, 4)
        plt.title("Location(Pred)")
        plt.imshow(loc_binary, cmap='gray')
        plt.colorbar()
        
    if diff_res is not None:
        plt.subplot(3, 3, 5)
        plt.title("Diff(Pred)")
        plt.imshow(diff_res, cmap='gray') # cmap='seismic' or 'bwr' might be better for diff if centered at 0, but gray is safe
        plt.colorbar()
        
    if final_res is not None:
        plt.subplot(3, 3, 6)
        plt.title("Final(Pred)")
        plt.imshow(final_res, cmap='gray') # cmap='seismic' or 'bwr' might be better for diff if centered at 0, but gray is safe
        plt.colorbar()
        
    if input_hu_array is not None:
        plt.subplot(3, 3, 7)
        plt.title("Input")
        plt.imshow(input_hu_array, cmap='gray') # cmap='seismic' or 'bwr' might be better for diff if centered at 0, but gray is safe
        plt.colorbar()
        
    if final_sythetic is not None:
        plt.subplot(3, 3, 8)
        plt.title("Final Synthetic")
        plt.imshow(final_sythetic, cmap='gray') # cmap='seismic' or 'bwr' might be better for diff if centered at 0, but gray is safe
        plt.colorbar()
        
    if final_gt is not None:
        plt.subplot(3, 3, 9)
        plt.title("Final GT")
        plt.imshow(final_gt, cmap='gray') # cmap='seismic' or 'bwr' might be better for diff if centered at 0, but gray is safe
        plt.colorbar()
        
    vis_path = os.path.join(output_dir, 'all_outputs.png')
    plt.tight_layout()
    plt.savefig(vis_path)
    print(f"Saved visualization to {vis_path}")
    plt.close()
    
def dicom_to_hu(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    pixel_array = ds.pixel_array
    hu_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    return hu_array

if __name__ == "__main__":
    
    # Default paths
    input_dicom_dir = "/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN/Kyunghee_Univ_Masked_10"
    input_dicom_file = f'{input_dicom_dir}/KP-0054/POST VUE/0150.dcm'
    gt_dicom_file = f'{input_dicom_dir}/KP-0054/POST STD/0150.dcm'
    input_hu_array = dicom_to_hu(input_dicom_file)
    gt_hu_array = dicom_to_hu(gt_dicom_file)
    input_file = './data/train/feature/KP-0054_0150.npy'
    gt_diff_file = f'./data/train/diff/KP-0054_0150.npy'
    gt_location_file = f'./data/train/location/KP-0054_0150.npy'
    
    # Checkpoints
    loc_ckpt = './decoder_location/checkpoints/best_model.pth'
    diff_ckpt = './decoder_diff/checkpoints/best_model.pth'
    
    output_dir = './decoder_test_results'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)

    # Validate checkpoints
    if not os.path.exists(loc_ckpt):
        print(f"Warning: Location checkpoint {loc_ckpt} not found.")
    if not os.path.exists(diff_ckpt):
        print(f"Warning: Diff checkpoint {diff_ckpt} not found.")

    test_decoders(input_file, loc_ckpt, diff_ckpt, output_dir, input_hu_array, gt_hu_array, gt_diff_file, gt_location_file)
