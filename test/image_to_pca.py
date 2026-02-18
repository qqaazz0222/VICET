import os
import urllib
from typing import Optional
import pydicom

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

# Constants
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)
PATCH_SIZE = 16
IMAGE_SIZE = 768
# IMAGE_SIZE = 768 * 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

def load_model(model_name: str, weights_path: str, device: str = "cuda"):
    """DINOv3 모델을 로드하고 로컬 가중치를 적용합니다."""
    print(f"Loading DINOv3 model: {model_name}")
    
    # Load model structure without downloading weights
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=model_name,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        pretrained=False
    )

    # Load local weights
    if os.path.exists(weights_path):
        print(f"Loading local weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    model.to(device)
    model.eval()
    return model

def load_image(uri: str) -> Image.Image:
    """URL 또는 로컬 파일 경로에서 이미지를 로드합니다."""
    if uri.startswith(('http://', 'https://')):
        with urllib.request.urlopen(uri) as f:
            image = Image.open(f).convert("RGB")
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
    elif uri.endswith('.dcm'):
        image, mask = load_dicom_image(uri)
    else:
        image = Image.open(uri).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
    return image, mask
    
def load_dicom_image(path: str, is_windowing: bool = False, upscale_ratio: int = 1, window_center: int = 40, window_width: int = 400) -> Image.Image:
    """DICOM 이미지를 로드하고 윈도잉을 적용하여 PIL Image로 변환합니다."""
    
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
        # -150~250 범위에 가중치를 더 주기 위한 piecewise linear mapping
        # 예: [-1024, -150, 250, max] -> [0, 30, 225, 255]
        # 이렇게 하면 -150~250(400 HU) 구간이 195 단계의 밝기를 차지하게 되어 대조도가 높아짐
        
        hu_points = [img_min, -150, 250, img_max]
        # 포인트들이 정렬되어 있어야 하므로 처리
        hu_points = sorted(list(set(hu_points)))
        
        # 각 HU 포인트에 대응하는 0-255 사이의 출력 값 설정
        # -150 미만은 0~30, -150~250 사이는 30~225, 250 초과는 225~255 할당
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
    pixel_array = raw_to_hu(ds).astype(np.float32)
    mask = np.zeros(pixel_array.shape, dtype=np.uint8)
    mask[pixel_array <= pixel_array[0, 0] + 64] = 1
    
    pixel_array = cut_air(pixel_array)
    if is_windowing:
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = windowing(pixel_array, window_center, window_width)
    else:
        img_min = pixel_array.min()
        img_max = pixel_array.max()
    
    original_pixel_array = pixel_array.copy()
        
    pixel_array = normalize(pixel_array, img_min, img_max)
    
    w, h = pixel_array.shape
    
    # Upscale
    if upscale_ratio > 1:
        pixel_array = cv2.resize(pixel_array, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_NEAREST)
    
    # PIL Image로 변환 (RGB)
    image = Image.fromarray(pixel_array).convert("RGB")
    return image, mask

def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """이미지 크기를 조정하고 정규화합니다."""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    
    image_resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    image_tensor = TF.to_tensor(image_resized)
    image_norm = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    return image_norm

def extract_features(model, image_tensor: torch.Tensor, model_name: str, device: str = "cuda"):
    """모델을 통해 이미지를 전달하고 중간 레이어의 특징을 추출합니다."""
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    
    with torch.inference_mode():
        with torch.autocast(device_type=device if "cuda" in device else "cpu", dtype=torch.float32):
            image_batch = image_tensor.unsqueeze(0).to(device)
            feats = model.get_intermediate_layers(image_batch, n=range(n_layers), reshape=True, norm=True)
            
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    return x

def compute_pca(features: torch.Tensor, h_patches: int, w_patches: int, mask_patches: Optional[np.ndarray] = None):
    """추출된 특징에 대해 PCA를 수행합니다. mask_patches가 1인 부분은 무시합니다."""
    features_np = features.numpy()
    
    # 마스크가 있으면 유효한 패치만 추출하여 PCA 피팅
    if mask_patches is not None:
        mask_flat = mask_patches.flatten()
        # mask_flat == 0 인 부분(배경이 아닌 부분)만 선택
        valid_indices = np.where(mask_flat == 0)[0]
        
        if len(valid_indices) > 0:
            pca = PCA(n_components=3, whiten=True)
            pca.fit(features_np[valid_indices])
            pca_features = pca.transform(features_np)
        else:
            print("Warning: All patches are masked. Falling back to all patches.")
            pca = PCA(n_components=3, whiten=True)
            pca_features = pca.fit_transform(features_np)
    else:
        pca = PCA(n_components=3, whiten=True)
        pca_features = pca.fit_transform(features_np)
    
    projected_image = torch.from_numpy(pca_features).view(h_patches, w_patches, 3)
    
    # Sigmoid를 적용하여 선명한 색상 생성
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # 결과에서도 마스크된 부분을 0으로 설정
    if mask_patches is not None:
        mask_torch = torch.from_numpy(mask_patches).unsqueeze(0)
        projected_image = projected_image * (1 - mask_torch)
        
    return projected_image

def visualize_results(original_image: Image.Image, pca_image: torch.Tensor, save_path: Optional[str] = None):
    """원본 이미지와 PCA 결과를 시각화합니다."""
    plt.figure(figsize=(15, 5), dpi=300)
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pca_image.permute(1, 2, 0))
    plt.axis('off')
    plt.title("PCA Visualisation")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pca_image.mean(dim=0), cmap='gray')
    plt.axis('off')
    plt.title("PCA Visualisation")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Result saved to {save_path}")

def main():
    # 설정
    MODEL_NAME = "dinov3_vitl16"
    WEIGHTS_PATH = "./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    IMAGE_PATH = "./test_dicom.dcm"
    SAVE_PATH = "./test_result.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 모델 로드
    model = load_model(MODEL_NAME, WEIGHTS_PATH, DEVICE)

    # 2. 이미지 로드 및 전처리
    image, mask = load_image(IMAGE_PATH)
    image_tensor = preprocess_image(image)
    
    # 패치 크기 계산
    h_patches = image_tensor.shape[1] // PATCH_SIZE
    w_patches = image_tensor.shape[2] // PATCH_SIZE
    
    # 3. 특징 추출
    features = extract_features(model, image_tensor, MODEL_NAME, DEVICE)

    # 마스크를 패치 크기에 맞춰 다운샘플링 (0 또는 1 유지를 위해 interpolation은 NEAREST 사용)
    mask_patches = cv2.resize(mask, (w_patches, h_patches), interpolation=cv2.INTER_NEAREST)

    # 4. PCA 계산 (마스크 전달)
    pca_image = compute_pca(features, h_patches, w_patches, mask_patches)
    
    print(f"Image Size: {image.size}")
    print(f"Image Patches: {w_patches}x{h_patches}")
    print(f"Features Shape: {features.shape}")
    print(f"PCA Image Shape: {pca_image.shape}")

    # 5. 시각화 및 저장
    visualize_results(image, pca_image, SAVE_PATH)

if __name__ == "__main__":
    main()