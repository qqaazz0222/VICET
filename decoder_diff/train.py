import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import DiffDataset
from model import RefinedDiffDecoder
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    data_dir = args.data_dir
    feature_dir = os.path.join(data_dir, 'feature')
    diff_dir = os.path.join(data_dir, 'diff')
    
    print(f"Loading data from {data_dir}...")
    dataset = DiffDataset(feature_dir, diff_dir)
    print(f"Total samples: {len(dataset)}")
    
    # Split
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = RefinedDiffDecoder().to(device)
    
    # Loss & Optimizer
    criterion = nn.L1Loss() # Changed to L1 Loss for sharper results
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train Loop
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            features, targets = data
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss /= train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                features, targets = data
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)
                
        val_loss /= val_size
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/ViTN2C/data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
