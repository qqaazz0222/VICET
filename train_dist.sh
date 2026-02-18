#!/bin/bash

SESSION_NAME="decoder_training"
EPOCHS=500
BATCH_SIZE=64
DATA_DIR="/workspace/Contrast_CT/hyunsu/VICET/data/train"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
  echo "Session '$SESSION_NAME' already exists."
  read -p "Do you want to kill the existing session and start a new one? (y/n): " choice
  case "$choice" in 
    y|Y ) 
      echo "Killing existing session..."
      tmux kill-session -t $SESSION_NAME
      sleep 5 # Wait for session cleanup
      ;;
    n|N ) 
      echo "Aborting. Attach to existing session with: tmux attach -t $SESSION_NAME"
      exit 0
      ;;
    * ) 
      echo "Invalid input. Aborting."
      exit 1
      ;;
  esac
fi

# Create new session
tmux new-session -d -s $SESSION_NAME -n 'diff'
if [ $? != 0 ]; then
  echo "Error creating tmux session. Please try again."
  exit 1
fi

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Window 0: Decoder Diff Training (GPUs 0,1,2,3)
tmux send-keys -t $SESSION_NAME:0 "cd decoder_diff" C-m
tmux send-keys -t $SESSION_NAME:0 "conda activate vicet" C-m
tmux send-keys -t $SESSION_NAME:0 "export MASTER_ADDR=localhost" C-m
tmux send-keys -t $SESSION_NAME:0 "export MASTER_PORT=29505" C-m
tmux send-keys -t $SESSION_NAME:0 "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t $SESSION_NAME:0 "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_addr=localhost --master_port=29505 train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --save_dir ./checkpoints 2>&1 | tee ../logs/diff_train_${TIMESTAMP}.log" C-m

sleep 5

# Window 1: Decoder Location Training (GPUs 4,5,6,7)
tmux new-window -t $SESSION_NAME -n 'location'
tmux send-keys -t $SESSION_NAME:1 "cd decoder_location" C-m
tmux send-keys -t $SESSION_NAME:1 "conda activate vicet" C-m
tmux send-keys -t $SESSION_NAME:1 "export MASTER_ADDR=localhost" C-m
tmux send-keys -t $SESSION_NAME:1 "export MASTER_PORT=29605" C-m
tmux send-keys -t $SESSION_NAME:1 "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t $SESSION_NAME:1 "CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_addr=localhost --master_port=29605 train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --save_dir ./checkpoints 2>&1 | tee ../logs/location_train_${TIMESTAMP}.log" C-m

echo "Training started in tmux session '$SESSION_NAME'"
echo "Diff Decoder is running on GPUs 0-3 (Port 29500)"
echo "Location Decoder is running on GPUs 4-7 (Port 29501)"
echo "Attach to session with: tmux attach -t $SESSION_NAME"
