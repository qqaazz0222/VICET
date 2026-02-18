import os
import subprocess

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error executing: {cmd}")
        return False
    return True

def verify():
    # 1. Verify decoder_diff training
    print("=== Verifying decoder_diff Training ===")
    cmd = "python3 /workspace/ViTN2C/decoder_diff/train.py --epochs 1 --batch_size 2 --save_dir /workspace/ViTN2C/decoder_diff/checkpoints_test"
    if not run_command(cmd): return
    
    # 2. Verify decoder_diff inference
    print("=== Verifying decoder_diff Inference ===")
    cmd = "python3 /workspace/ViTN2C/decoder_diff/inference.py --input_path /workspace/ViTN2C/data/feature/KP-0010_0001.npy --checkpoint /workspace/ViTN2C/decoder_diff/checkpoints_test/best_model.pth --output_dir /workspace/ViTN2C/decoder_diff/output_test"
    if not run_command(cmd): return

    # 3. Verify decoder_location training
    print("=== Verifying decoder_location Training ===")
    cmd = "python3 /workspace/ViTN2C/decoder_location/train.py --epochs 1 --batch_size 2 --save_dir /workspace/ViTN2C/decoder_location/checkpoints_test"
    if not run_command(cmd): return

    # 4. Verify decoder_location inference
    print("=== Verifying decoder_location Inference ===")
    cmd = "python3 /workspace/ViTN2C/decoder_location/inference.py --input_path /workspace/ViTN2C/data/feature/KP-0010_0001.npy --checkpoint /workspace/ViTN2C/decoder_location/checkpoints_test/best_model.pth --output_dir /workspace/ViTN2C/decoder_location/output_test"
    if not run_command(cmd): return
    
    print("\nALL VERIFICATIONS PASSED!")

if __name__ == "__main__":
    verify()
