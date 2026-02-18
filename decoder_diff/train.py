
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
from dataset import DiffDataset
from model import RefinedDiffDecoder
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def train(args):
    # Setup DDP
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Using device: {device}")
        
    # Dataset
    data_dir = args.data_dir
    feature_dir = os.path.join(data_dir, 'feature')
    diff_dir = os.path.join(data_dir, 'diff')
    
    if rank == 0:
        print(f"Loading data from {data_dir}...")
    
    # Ensure dataset creation is consistent across ranks
    dataset = DiffDataset(feature_dir, diff_dir)
    if rank == 0:
        print(f"Total samples: {len(dataset)}")
    
    # Split
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    
    # Use a fixed generator for random_split to ensure same split on all ranks
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    if rank == 0:
        print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # Model
    model = RefinedDiffDecoder().to(device)
    # Convert BatchNorm to SyncBatchNorm if present (optional but recommended)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Loss & Optimizer
    criterion = nn.L1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train Loop
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume Logic
    last_ckpt_path = os.path.join(args.save_dir, 'last_model.pth')
    if os.path.exists(last_ckpt_path):
        if rank == 0:
            print(f"Found checkpoint at {last_ckpt_path}. Resuming...")
        # Map location to the correct device
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        
        # Check if checkpoint is a dict with detailed info or just state_dict (legacy)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            if rank == 0:
                print(f"Resumed from epoch {start_epoch}, Best Loss: {best_loss:.4f}")
        else:
            # Fallback for simple state_dict checkpoints
            model.module.load_state_dict(checkpoint)
            if rank == 0:
                print("Loaded legacy checkpoint (weights only). Starting from epoch 0.")
    
    if rank == 0:
        print(f"Starting training from epoch {start_epoch+1}...")
        
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
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
            
        # Aggregate loss across all ranks
        total_train_loss_tensor = torch.tensor(train_loss).to(device)
        dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
        # Average over total dataset size (train_size)
        epoch_train_loss = total_train_loss_tensor.item() / train_size
        
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
        
        total_val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = total_val_loss_tensor.item() / val_size

        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(model.module.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
            
            # Save last model (Full Checkpoint)
            last_save_path = os.path.join(args.save_dir, 'last_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, last_save_path)
                
    cleanup_distributed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default data dir updated to match user context roughly, but argparse allows overriding
    parser.add_argument('--data_dir', type=str, default='/workspace/Contrast_CT/hyunsu/VICET/data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if "LOCAL_RANK" in os.environ:
        train(args)
    else:
        print("Error: Please run with torchrun. Example:")
        print(f"torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) {__file__} --data_dir {args.data_dir}")
