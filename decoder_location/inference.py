import os
import glob
import argparse
import torch
import numpy as np
from model import RefinedLocationDecoder

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = RefinedLocationDecoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Process
    if os.path.isdir(args.input_path):
        files = sorted(glob.glob(os.path.join(args.input_path, "*.npy")))
    else:
        files = [args.input_path]
        
    print(f"Found {len(files)} files to process.")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for f_path in files:
            basename = os.path.basename(f_path)
            
            # Load Feature
            try:
                feature = np.load(f_path)
                feature = feature.reshape(48, 48, 1024).transpose(2, 0, 1)
                feature = torch.from_numpy(feature).float().unsqueeze(0).to(device)
                
                # Infer
                outputs = model(feature)
                outputs = torch.sigmoid(outputs) # Apply sigmoid for location
                outputs = (outputs > 0.5).float() # Binarize for clearer output
                outputs = outputs.cpu().numpy().squeeze()
                
                # Save
                save_path = os.path.join(args.output_dir, basename)
                np.save(save_path, outputs)
                print(f"Saved {save_path}")
            except Exception as e:
                print(f"Error processing {f_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Path to feature file or directory")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()
    
    inference(args)
