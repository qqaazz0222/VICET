# VICET (Virtual Contrast Enhancement Transformer)

## 프로젝트 소개 (Introduction)
VICET는 DINOv3 기반의 Transformer 모델을 활용하여 Non-Contrast CT (NCCT) 영상으로부터 Contrast-Enhanced CT (CECT)와 유사한 조영 증강 효과를 가상으로 생성하는 딥러닝 프로젝트입니다. 
이 모델은 입력 영상에서 추출한 고차원 특징(Feature)을 바탕으로 조영제가 투여된 것과 같은 시각적 효과를 합성하여 진단 보조 등의 목적으로 활용될 수 있습니다.

## 주요 기능 (Key Features)
- **Feature Extraction**: DINOv3 (Vision Transformer)를 사용하여 NCCT 영상에서 강력한 의미론적 특징을 추출합니다.
- **Diff Decoder**: 추출된 특징으로부터 원본 영상과 조영 증강 영상 간의 차이(Diff Map)를 예측합니다.
- **Location Decoder**: 조영 증강이 발생할 해부학적 위치(Location Map)를 예측하여 불필요한 영역의 노이즈를 억제합니다.
- **Image Synthesis**: 예측된 Diff Map과 Location Map을 결합하고 후처리를 적용하여 최종 가상 CECT 영상을 생성합니다.

## 디렉토리 구조 (Directory Structure)
```
VICET/
├── data/
│   ├── dicom/            # 원본 DICOM 파일 (NCCT, CECT) 저장
│   ├── feature/          # DINOv3로 추출된 특징 파일 (.npy)
│   ├── diff/             # Ground Truth 차분 맵 (.npy)
│   └── location/         # Ground Truth 위치 맵 (.npy)
├── decoder_diff/         # Diff Decoder 모델 및 학습 관련 코드
│   ├── model.py          # 모델 아키텍처 정의
│   ├── train.py          # 학습 스크립트
│   └── checkpoints/      # 학습된 가중치 저장
├── decoder_location/     # Location Decoder 모델 및 학습 관련 코드
│   ├── model.py          # 모델 아키텍처 정의
│   ├── train.py          # 학습 스크립트
│   └── checkpoints/      # 학습된 가중치 저장
├── dinov3/               # DINOv3 모델 관련 라이브러리
├── dicom_to_feature.py   # 데이터 전처리 및 특징 추출 스크립트
├── test_decoder.py       # 추론 및 결과 시각화 스크립트
├── train.sh              # 전체 모델 학습 실행 스크립트
└── verify.py             # 데이터 검증 스크립트
```

## 설치 및 요구사항 (Installation)
본 프로젝트는 Python 3.8 이상 및 PyTorch 환경에서 동작합니다. Conda를 사용하여 가상 환경을 설정하는 것을 권장합니다.

```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate vicet
```

학습 스크립트(`train.sh`) 실행을 위해 `tmux`가 설치되어 있어야 합니다. (Linux 환경 권장)
```bash
sudo apt-get install tmux
```

## 사용 방법 (Usage)

### 1. 데이터 준비 (Data Preparation)
DICOM 파일을 처리하여 학습에 필요한 DINOv3 특징과 Ground Truth 맵을 생성합니다.
`dicom_to_feature.py`를 실행하기 전에 `main` 함수 내의 `generate_feature` 함수 호출 부분의 주석을 해제해야 입력 특징을 생성할 수 있습니다.

```bash
# dicom_to_feature.py 실행
python dicom_to_feature.py --dicom_dir ./data/dicom --feature_dir ./data/feature
```
- `--dicom_dir`: 환자별 DICOM 폴더가 있는 경로
- `--feature_dir`: 추출된 특징을 저장할 경로

### 2. 모델 학습 (Training)
`train.sh` 스크립트를 사용하여 Diff Decoder와 Location Decoder를 병렬로 학습합니다. 이 스크립트는 `tmux` 세션을 생성하여 두 모델을 동시에 학습합니다.

**주의사항**: `train.sh` 파일 내의 경로(`/workspace/ViTN2C/...`)가 현재 프로젝트 경로와 일치하는지 확인하고 필요 시 수정하세요.

```bash
chmod +x train.sh
./train.sh
```

### 3. 추론 및 시각화 (Inference & Visualization)
학습된 모델을 사용하여 단일 케이스에 대한 추론을 수행하고 결과를 시각화합니다.
`test_decoder.py` 파일의 하단 `if __name__ == "__main__":` 블록에서 테스트할 파일 경로와 체크포인트 경로를 자신의 환경에 맞게 수정해야 합니다.

```bash
python test_decoder.py
```
실행 결과는 `decoder_test_results/` 디렉토리에 `all_outputs.png` 이미지와 결과 `.npy` 파일들로 저장됩니다.

## 모델 아키텍처 (Model Architecture)
- **Encoder**: DINOv3 (ViT-L/16 등)를 사용하여 이미지를 임베딩합니다.
- **RefinedDiffDecoder**: ResNet 스타일의 ConvBlock과 Upsampling 레이어를 통해 1024채널의 특징을 512x512 크기의 Diff Map으로 복원합니다.
- **RefinedLocationDecoder**: DiffDecoder와 유사한 구조를 가지며, 조영 증강 영역을 이진화하여 예측합니다.

## 라이선스 (License)
MIT License
