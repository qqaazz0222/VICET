import os
import glob
import argparse
from tqdm import tqdm
from typing import Optional
import numpy as np
import pydicom
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import functools

def log_execution(func):
    """
    함수의 실행 시작과 종료를 출력하는 데코레이터입니다.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 함수 이름을 대문자로 변환하여 시작 로그 출력
        print(f"\033[1m[>] Starting {func.__name__.replace('_', ' ').title()}...\033[0m")
        result = func(*args, **kwargs)
        # 성공적으로 완료되었음을 알리는 로그 출력
        print(f"\033[1;32m[+] Finished {func.__name__.replace('_', ' ').title()}!\033[0m")
        return result
    return wrapper

def log_tab(string: str, tab_count: int = 1):
    """
    지정한 탭 횟수만큼 들여쓰기하여 문자열을 출력합니다.
    """
    print(f"    " * tab_count + string)

@log_execution
def init_args():
    """
    명령행 인자(Arguments)를 초기화하고 필요한 디렉토리를 생성합니다.
    """
    parser = argparse.ArgumentParser(description="DICOM 이미지를 처리하고 특징을 추출하는 스크립트")
    
    # 기본 경로 설정
    # parser.add_argument("--dicom_dir", type=str, default="./data/dicom", help="입력 DICOM 파일이 위치한 디렉토리")
    parser.add_argument("--dicom_dir", type=str, default="/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN/Kyunghee_Univ_Masked_10", help="입력 DICOM 파일이 위치한 디렉토리")
    parser.add_argument("--train_dir", type=str, default="./data/train", help="입력 DICOM 파일이 위치한 디렉토리")
    parser.add_argument("--feature_dir", type=str, default="./data/train/feature", help="추출된 특징(.npy)을 저장할 디렉토리")
    parser.add_argument("--diff_dir", type=str, default="./data/train/diff", help="차분 맵(Difference Map)을 저장할 디렉토리")
    parser.add_argument("--location_dir", type=str, default="./data/train/location", help="위치 맵(Location Map)을 저장할 디렉토리")
    
    # 데이터 상세 설정
    parser.add_argument("--ncct_dir", type=str, default="POST VUE", help="NCCT 시리즈를 식별하기 위한 문자열")
    parser.add_argument("--cect_dir", type=str, default="POST STD", help="CECT 시리즈를 식별하기 위한 문자열")
    
    # 모델 설정
    parser.add_argument("--model_name", type=str, default="dinov3_vitl16", help="사용할 DINOv3 모델 이름")
    parser.add_argument("--weights_path", type=str, default="./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", help="로컬 모델 가중치 경로")
    parser.add_argument("--patch_size", type=int, default=16, help="DINOv3의 패치 크기")
    parser.add_argument("--image_size", type=int, default=768, help="입력 이미지 리사이즈 크기")
    
    # 정규화 설정 (ImageNet 기준)
    parser.add_argument("--imagenet_mean", type=float, default=(0.485, 0.456, 0.406), help="정규화 시 사용할 평균값")
    parser.add_argument("--imagenet_std", type=float, default=(0.229, 0.224, 0.225), help="정규화 시 사용할 표준편차값")
    
    parser.add_argument("--device", type=str, default="cuda", help="연산에 사용할 장치 (cuda 또는 cpu)")
    
    args = parser.parse_args()
    
    # 입력 디렉토리 존재 확인 및 필요한 출력 디렉토리 생성
    if not os.path.exists(args.dicom_dir):
        os.makedirs(args.dicom_dir, exist_ok=True)
        raise FileNotFoundError(f"입력 디렉토리 {args.dicom_dir}를 찾을 수 없습니다.")
    
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)
    os.makedirs(args.diff_dir, exist_ok=True)
    os.makedirs(args.location_dir, exist_ok=True)
    
    return args

@log_execution
def load_model(model_name: str, weights_path: str, device: str = "cuda"):
    """
    DINOv3 모델을 로컬 디렉토리에서 로드하고 가중치를 적용합니다.
    """
    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
    # DINOV3_LOCATION 환경변수가 있으면 해당 경로를 사용하고, 없으면 GitHub를 기본값으로 함
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)
    
    # torch.hub를 이용해 모델 로드
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=model_name,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        pretrained=False
    )

    # 로컬 가중치 파일이 존재하면 로드
    if os.path.exists(weights_path):
        log_tab(f"Loading local weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")

    # 모델을 연산 장치로 이동 및 평가 모드 설정
    model.to(device)
    model.eval()
    return model

@log_execution
def load_dataset(input_dir: str, ncct_dir: str = "POST VUE", cect_dir: str = "POST STD"):
    """
    DICOM 데이터셋을 탐색하여 환자별 NCCT 및 CECT 파일 목록을 구성합니다.
    """
    dataset = {"ncct":{}, "cect":{}}
    
    # 환자 아이디별 디렉토리 목록 정렬
    patient_list = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for patient_id in tqdm(patient_list, desc="    Loading dataset"):
        # 환자 디렉토리 내의 모든 DICOM 파일 재귀적 검색
        dcm_file_list = sorted(glob.glob(os.path.join(input_dir, patient_id, "**", "*.dcm"), recursive=True))
        
        # 특정 키워드를 포함하는 파일 리스트 필터링 (NCCT vs CECT)
        ncct_dcm_file_list = [dcm_file for dcm_file in dcm_file_list if ncct_dir in dcm_file]
        cect_dcm_file_list = [dcm_file for dcm_file in dcm_file_list if cect_dir in dcm_file]
        
        # InstanceNumber를 기준으로 파일 정렬 (슬라이스 순서 보장)
        sorted_ncct_dcm_file_list = sorted(ncct_dcm_file_list, key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        sorted_cect_dcm_file_list = sorted(cect_dcm_file_list, key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        
        dataset["ncct"][patient_id] = sorted_ncct_dcm_file_list
        dataset["cect"][patient_id] = sorted_cect_dcm_file_list
        
    return dataset
    
def load_dicom_image(path: str, is_raw: bool = False, is_windowing: bool = False, upscale_ratio: int = 1, window_center: int = 40, window_width: int = 400):
    """
    DICOM 파일을 로드하여 픽셀 값 조정, 윈도잉, 정규화를 거쳐 PIL 이미지로 반환합니다.
    """
    
    def raw_to_hu(ds):
        # DICOM 태그를 이용하여 Raw 데이터를 HU(Hounsfield Unit)로 변환
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            return ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        else:
            return ds.pixel_array
        
    def cut_air(pixel_array):
        # 공기 부문의 노이즈를 제어하기 위해 최소 HU 값을 -1024로 고정
        image_min = -1024
        image_max = pixel_array.max()
        pixel_array = np.clip(pixel_array, image_min, image_max)
        return pixel_array
        
    def windowing(pixel_array, window_center, window_width):
        # 특정 장기나 조직을 잘 보기 위해 윈도잉(밝기/대비 조정) 적용
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, img_min, img_max)
        return pixel_array
    
    def normalize(pixel_array, img_min, img_max):
        # 비선형 인터폴레이션을 통한 0~255 범위 정규화 (전처리 기법)
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
    
    # DICOM 파일 읽기
    ds = pydicom.dcmread(path)
    pixel_array = raw_to_hu(ds)
    
    # 주변 배경(Air)을 마스킹하기 위한 간단한 마스크 생성
    mask = np.zeros(pixel_array.shape, dtype=np.uint8)
    mask[pixel_array <= pixel_array[0, 0] + 64] = 1
    
    # 원본 HU 데이터가 필요한 경우 처리
    if is_raw:
        return pixel_array, mask
    
    # 전처리 단계: 공기 부문 절단 및 선택적 윈도잉
    pixel_array = cut_air(pixel_array)
    if is_windowing:
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = windowing(pixel_array, window_center, window_width)
    else:
        img_min = pixel_array.min()
        img_max = pixel_array.max()
        
    # 정규화 수행
    pixel_array = normalize(pixel_array, img_min, img_max)
    
    # 해상도 확대(Upscaling)가 필요한 경우 처리
    if upscale_ratio > 1:
        pixel_array = cv2.resize(pixel_array, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_NEAREST)
    
    # PIL 이미지 형식으로 변환하여 반환
    image = Image.fromarray(pixel_array).convert("RGB")
    return image, mask

def preprocess_image(image: Image.Image, image_size: int, patch_size: int, imagenet_mean: tuple, imagenet_std: tuple) -> torch.Tensor:
    """
    PIL 이미지를 모델 입력 규격에 맞게 리사이즈하고 텐서로 변환 및 정규화합니다.
    """
    w, h = image.size
    # 패치 크기에 맞게 최종 이미지 크기 계산
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    
    # 리사이즈 및 정규화
    image_resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    image_tensor = TF.to_tensor(image_resized)
    image_norm = TF.normalize(image_tensor, mean=imagenet_mean, std=imagenet_std)
    
    return image_norm

def extract_features(model, image_tensor: torch.Tensor, model_name: str, device: str = "cuda"):
    """
    사전 학습된 모델을 통해 이미지의 중간 레이어 특징(Feature)을 추출합니다.
    """
    # 각 모델별 레이어 깊이 설정
    model_to_num_layers = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    
    n_layers = model_to_num_layers[model_name]
    
    with torch.inference_mode():
        # 혼합 정밀도(Mixed Precision) 사용 여부 설정
        with torch.autocast(device_type=device if "cuda" in device else "cpu", dtype=torch.float32):
            image_batch = image_tensor.unsqueeze(0).to(device)
            # DINOv3의 모든 중간 레이어 특징 추출
            feats = model.get_intermediate_layers(image_batch, n=range(n_layers), reshape=True, norm=True)
            
            # 마지막 레이어의 특징을 가져와 정규화 및 변형
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    return x

@log_execution
def generate_feature(args, model, dataset):
    """
    주어진 데이터셋의 각 DICOM 이미지에 대해 특징을 추출하고 저장합니다.
    """
    for patient_id in tqdm(dataset.keys(), desc="    Processing patients"):
        dcm_file_list = dataset[patient_id]
        
        for dcm_file in dcm_file_list:
            # 이미지 로드 및 전처리
            image, mask = load_dicom_image(dcm_file)
            image_tensor = preprocess_image(image, args.image_size, args.patch_size, args.imagenet_mean, args.imagenet_std)
            
            # 특징 추출
            features = extract_features(model, image_tensor, args.model_name, args.device)
            features_np = features.numpy()
            
            # 결과 저장 (.npy 형식)
            save_path = os.path.join(args.feature_dir, f"{patient_id}_{os.path.basename(dcm_file).replace('.dcm', '.npy')}")
            np.save(save_path, features_np)
            
@log_execution
def generate_map_feature(args, model, ncct_dataset, cect_dataset, min_threshold=200, max_threshold=2000):
    """
    NCCT와 CECT 이미지 사이의 차분 특징(Difference Map) 및 조영 효과 위치를 생성합니다.
    """
    
    def smoothing_diff_map(diff_map, kernel_size=3, iterations=5):
        """
        0이 아닌 유효 픽셀 값을 기반으로 Dilation(팽창) 연산을 수행하여 주변의 0인 영역을 채웁니다.
        보존할 영역(200 HU 이상)의 경계를 부드럽게 블렌딩하여 합칩니다.
        """
        # 커널 생성
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 1. 보존할 영역 마스크 생성 (200 HU 이상)
        preservation_threshold = 160
        # 마스크를 float32로 변환 (0.0 ~ 1.0)
        preservation_mask = (diff_map >= preservation_threshold).astype(np.float32)
        
        # 2. 마스크 경계 블러링 (Soft Mask 생성)
        # 커널 크기는 홀수여야 함
        blur_kernel_size = kernel_size * 2 + 1
        soft_mask = cv2.GaussianBlur(preservation_mask, (blur_kernel_size, blur_kernel_size), 0)
        
        # 3. 배경 생성 (Dilation + Blur)
        # Dilation 수행: 0이 아닌 밝은 값들이 주변으로 확장됨
        dilated_map = cv2.dilate(diff_map, kernel, iterations=iterations)
        # dilated_map에 가우시안 블러 적용하여 부드럽게 만듦
        dilated_map = cv2.GaussianBlur(dilated_map, (blur_kernel_size, blur_kernel_size), 0)
    
        # 4. 알파 블렌딩
        # soft_mask가 1에 가까운 곳은 원본(diff_map)을, 0에 가까운 곳은 배경(dilated_map)을 사용
        # 경계 부분은 두 값이 섞이게 됨
        filtered_map = (diff_map * soft_mask) + (dilated_map * (1 - soft_mask))
        
        return filtered_map

    for patient_id in tqdm(ncct_dataset.keys(), desc="    Processing patients"):
        ncct_dcm_file_list = ncct_dataset[patient_id]
        cect_dcm_file_list = cect_dataset[patient_id]
        
        # 슬라이스별로 순회하며 차분 연산 수행
        for ncct_dcm_file, cect_dcm_file in zip(ncct_dcm_file_list, cect_dcm_file_list):
            ncct_image, _ = load_dicom_image(ncct_dcm_file, is_raw=True)
            cect_image, _ = load_dicom_image(cect_dcm_file, is_raw=True)
            
            # 차분 계산 (CECT - NCCT)
            diff_map = cect_image - ncct_image
            # 특정 임계치(Threshold)를 기준으로 값 클리핑
            diff_map[diff_map < 0] = 0
            diff_map[diff_map > max_threshold] = 0
            
            # 조영 증가가 일어난 영역(Location Map) 생성
            location_map = np.zeros(diff_map.shape, dtype=np.uint8)
            location_map[diff_map >= min_threshold] = 1

            # diff_map에 smoothing_diff_map 적용
            # diff_map = smoothing_diff_map(diff_map)
            
            # 결과 저장
            diff_save_path = os.path.join(args.diff_dir, f"{patient_id}_{os.path.basename(ncct_dcm_file).replace('.dcm', '.npy')}")
            location_save_path = os.path.join(args.location_dir, f"{patient_id}_{os.path.basename(ncct_dcm_file).replace('.dcm', '.npy')}")
            np.save(diff_save_path, diff_map)
            np.save(location_save_path, location_map)  

def main():
    """
    데이터 준비, 모델 로드, 특징 추출 및 저장의 전체 파이프라인을 실행합니다.
    """
    # 1. 인자 초기화
    args = init_args()
    # 2. 모델 로드
    model = load_model(args.model_name, args.weights_path, args.device)
    # 3. 데이터셋 구성
    dataset = load_dataset(args.dicom_dir, args.ncct_dir, args.cect_dir)
    
    log_tab(f"Number of patients: {len(dataset['ncct'])}")   
    
    # 4. 특징 추출 및 파일 저장
    # generate_feature(args, model, dataset["ncct"])
    generate_map_feature(args, model, dataset["ncct"], dataset["cect"])
    
    log_tab(f"Done! Features are saved to {args.feature_dir}")
    
if __name__ == "__main__":
    main()