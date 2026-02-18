
import os
import glob
import argparse
import sys
import numpy as np
import pydicom
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import importlib.util
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- Helper Functions from dicom_to_feature.py ---

def load_dicom_image(path: str, is_raw: bool = False, is_windowing: bool = False, upscale_ratio: int = 1, window_center: int = 40, window_width: int = 400):
    
    def raw_to_hu(ds):
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            return ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        else:
            return ds.pixel_array
        
    def cut_air(pixel_array):
        image_min = -1024
        image_max = pixel_array.max()
        pixel_array = np.clip(pixel_array, image_min, image_max)
        return pixel_array
        
    def windowing(pixel_array, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, img_min, img_max)
        return pixel_array
    
    def normalize(pixel_array, img_min, img_max):
        hu_points = [img_min, -150, 250, img_max]
        hu_points = sorted(list(set(hu_points)))
        
        out_points = []
        for p in hu_points:
            if p <= -150:
                val = 30 * (p - img_min) / (-150 - img_min + 1e-5) if p > img_min else 0
            elif p <= 250:
                val = 30 + (195 * (p - (-150)) / (250 - (-150)))
            else:
                val = 225 + (30 * (p - 250) / (img_max - 250 + 1e-5)) if p < img_max else 255
            out_points.append(val)

        pixel_array = np.interp(pixel_array, hu_points, out_points)
        pixel_array = pixel_array.astype(np.uint8)
        return pixel_array
    
    ds = pydicom.dcmread(path)
    pixel_array = raw_to_hu(ds)
    
    # Mask generation
    mask = np.zeros(pixel_array.shape, dtype=np.uint8)
    if pixel_array.size > 0:
        mask[pixel_array <= pixel_array[0, 0] + 64] = 1
    
    if is_raw:
        return pixel_array, mask
    
    pixel_array = cut_air(pixel_array)
    if is_windowing:
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = windowing(pixel_array, window_center, window_width)
    else:
        img_min = pixel_array.min()
        img_max = pixel_array.max()
        
    pixel_array = normalize(pixel_array, img_min, img_max)
    
    if upscale_ratio > 1:
        pixel_array = cv2.resize(pixel_array, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_NEAREST)
    
    image = Image.fromarray(pixel_array).convert("RGB")
    return image, mask

def preprocess_image(image: Image.Image, image_size: int, patch_size: int, imagenet_mean: tuple, imagenet_std: tuple) -> torch.Tensor:
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    
    image_resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    image_tensor = TF.to_tensor(image_resized)
    image_norm = TF.normalize(image_tensor, mean=imagenet_mean, std=imagenet_std)
    
    return image_norm

def load_dinov3_model(model_name: str, weights_path: str, device: str = "cuda"):
    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)
    
    print(f"Loading DINOv3 model: {model_name} from {DINOV3_LOCATION}...")
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=model_name,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        pretrained=False
    )

    if os.path.exists(weights_path):
        print(f"Loading local weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.to(device)
    model.eval()
    return model

def extract_dinov3_features(model, image_tensor: torch.Tensor, model_name: str, device: str = "cuda"):
    # Layer config
    model_to_num_layers = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    n_layers = model_to_num_layers.get(model_name, 24) 
    
    with torch.inference_mode():
        with torch.autocast(device_type=device if "cuda" in device else "cpu", dtype=torch.float32):
            image_batch = image_tensor.unsqueeze(0).to(device)
            # reshape=True gives (B, H, W, D) or (B, D, H, W)
            feats = model.get_intermediate_layers(image_batch, n=range(n_layers), reshape=True, norm=True)
            
            # Use last layer features
            feature_map = feats[-1] 
            
            # Ensure shape is (B, D, H, W)
            # ViT-L dim is 1024
            if feature_map.shape[-1] == 1024: # (B, H, W, D)
                feature_map = feature_map.permute(0, 3, 1, 2) # (B, D, H, W)
            # If feature_map.shape[1] == 1024, it is already (B, D, H, W)
            
    return feature_map

# --- Helper Functions from test_decoder.py ---

def load_module_from_path(path, module_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# --- Main Logic ---

def save_dicom(original_dcm_path, output_path, synthetic_hu, original_ds):
    """
    Save synthetic HU array as DICOM, preserving metadata from original.
    """
    # Convert HU back to pixel values
    # PixelValue = (HU - Intercept) / Slope
    intercept = original_ds.RescaleIntercept if hasattr(original_ds, 'RescaleIntercept') else 0
    slope = original_ds.RescaleSlope if hasattr(original_ds, 'RescaleSlope') else 1
    
    pixel_array = (synthetic_hu - intercept) / slope
    
    # Clip to valid range of original pixel representation if possible
    # Assuming original is typically uint16 or int16
    if original_ds.PixelRepresentation == 0: # unsigned
        pixel_array = np.clip(pixel_array, 0, 65535).astype(np.uint16)
    else: # signed
        pixel_array = np.clip(pixel_array, -32768, 32767).astype(np.int16)

    original_ds.PixelData = pixel_array.tobytes()
    original_ds.save_as(output_path)

import torch.multiprocessing as mp
import math

def process_chunk(gpu_id, file_chunk, args):
    """
    Process a chunk of files on a specific GPU.
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    # 1. Load Models per process
    # print(f"[GPU {gpu_id}] Loading models...")
    dinov3 = load_dinov3_model(args.model_name, args.dinov3_weights, device=f"cuda:{gpu_id}")
    
    LocationModel = load_module_from_path('./decoder_location/model.py', 'location_model')
    DiffModel = load_module_from_path('./decoder_diff/model.py', 'diff_model')
    
    loc_decoder = LocationModel.RefinedLocationDecoder().to(device)
    loc_decoder.load_state_dict(torch.load(args.loc_weights, map_location=device))
    loc_decoder.eval()
    
    diff_decoder = DiffModel.RefinedDiffDecoder().to(device)
    diff_decoder.load_state_dict(torch.load(args.diff_weights, map_location=device))
    diff_decoder.eval()
    
    # print(f"[GPU {gpu_id}] Models loaded. Processing {len(file_chunk)} files...")
    
    for dcm_path in tqdm(file_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            # Prepare Output Path
            rel_path = os.path.relpath(dcm_path, args.input_dir)
            output_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load Original
            ds = pydicom.dcmread(dcm_path)
            if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
                input_hu = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            else:
                input_hu = ds.pixel_array
            
            # Preprocess
            pil_img, _ = load_dicom_image(dcm_path)
            img_tensor = preprocess_image(pil_img, args.image_size, args.patch_size, args.imagenet_mean, args.imagenet_std)
            
            # Extract Features
            feature_map = extract_dinov3_features(dinov3, img_tensor, args.model_name, device=f"cuda:{gpu_id}")
            
            # Run Decoders
            with torch.no_grad():
                loc_out = loc_decoder(feature_map)
                loc_prob = torch.sigmoid(loc_out).cpu().numpy().squeeze()
                
                diff_out = diff_decoder(feature_map)
                diff_res = diff_out.cpu().numpy().squeeze()
                
            # Resize
            target_h, target_w = input_hu.shape
            if diff_res.shape != (target_h, target_w):
                diff_res = cv2.resize(diff_res, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                loc_prob = cv2.resize(loc_prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # Synthesize
            hu_loc_binary = np.zeros_like(input_hu)
            hu_loc_binary[input_hu > 16] = 1
            
            ratio = 0.15
            final_res = diff_res * hu_loc_binary * ratio + diff_res * loc_prob * (1 - ratio)
            final_res *= 2.5
            final_res = gaussian_filter(final_res, sigma=1.0)
            
            final_synthetic = input_hu + final_res
            final_synthetic = np.clip(final_synthetic, -1024, 3071)
            
            save_dicom(dcm_path, output_path, final_synthetic, ds)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {dcm_path}: {e}")
            continue

def run_multi_gpu(args):
    # 1. Find DICOM files
    print("Scanning DICOM files...")
    search_pattern = os.path.join(args.input_dir, "**", "*.dcm")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    
    # Filter for NCCT keyword
    input_files = [f for f in all_files if args.ncct_keyword in f] if args.ncct_keyword else all_files
    print(f"Found {len(input_files)} DICOM files matching keyword '{args.ncct_keyword}'.")
    
    # 2. Determine GPUs
    if args.gpus:
        gpu_list = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_list = list(range(torch.cuda.device_count()))
    
    print(f"Using GPUs: {gpu_list}")
    
    if len(gpu_list) == 0:
        print("No GPUs available!")
        return

    # 3. Split files into chunks
    num_gpus = len(gpu_list)
    chunk_size = math.ceil(len(input_files) / num_gpus)
    chunks = [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]
    
    # 4. Launch Processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for i, gpu_id in enumerate(gpu_list):
        if i < len(chunks):
            p = mp.Process(target=process_chunk, args=(gpu_id, chunks[i], args))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
    
    print("All tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VICET Pipeline: DICOM -> DINOv3 -> Decoders -> Synthetic DICOM (Multi-GPU)")
    
    parser.add_argument("--input_dir", type=str, default="/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN/Kyunghee_Univ_Masked", help="Root directory containing input DICOMs (NCCT)")
    parser.add_argument("--output_dir", type=str, default="./data/SYNTHETIC", help="Directory to save synthetic DICOMs")
    parser.add_argument("--ncct_keyword", type=str, default="POST VUE", help="Keyword to filter NCCT DICOM files")
    
    # Model Weights
    parser.add_argument("--dinov3_weights", type=str, default="./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--loc_weights", type=str, default="./decoder_location/checkpoints/best_model.pth")
    parser.add_argument("--diff_weights", type=str, default="./decoder_diff/checkpoints/best_model.pth")
    
    # DINOv3 Config
    parser.add_argument("--model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=768)
    parser.add_argument("--imagenet_mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--imagenet_std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). Default: all available.")
    
    args = parser.parse_args()
    
    # Convert lists to tuples for consistency
    args.imagenet_mean = tuple(args.imagenet_mean)
    args.imagenet_std = tuple(args.imagenet_std)
    
    run_multi_gpu(args)
