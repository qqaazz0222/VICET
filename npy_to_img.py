import numpy as np
import cv2
import os

def n2i(npy_path: str, img_path: str):
    npy_data = np.load(npy_path)
    npy_data = (npy_data - npy_data.min()) / (npy_data.max() - npy_data.min()) * 255
    npy_data = npy_data.astype(np.uint8)
    cv2.imwrite(img_path, npy_data)

if __name__ == "__main__":
    n2i("/workspace/ViTN2C/data/diff/KP-0010_0155.npy", "./debug_diff.jpg")
    n2i("/workspace/ViTN2C/data/location/KP-0010_0155.npy", "./debug_location.jpg")
    # n2i("/workspace/ViTN2C/decoder_diff/output_test/KP-0010_0155.npy", "./debug_diff_result.jpg")
    # n2i("/workspace/ViTN2C/decoder_location/output_test/KP-0010_0155.npy", "./debug_location_result.jpg")