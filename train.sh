#!/bin/bash

SESSION_NAME="decoder_training"
EPOCHS=500
BATCH_SIZE=24

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
  echo "Session '$SESSION_NAME' already exists."
  read -p "Do you want to kill the existing session and start a new one? (y/n): " choice
  case "$choice" in 
    y|Y ) 
      echo "Killing existing session..."
      tmux kill-session -t $SESSION_NAME
      sleep 1 # Wait for session cleanup
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

# Window 0: Decoder Diff Training
tmux send-keys -t $SESSION_NAME:0 "cd decoder_diff" C-m
tmux send-keys -t $SESSION_NAME:0 "python3 train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --save_dir ./checkpoints" C-m

# Window 1: Decoder Location Training
tmux new-window -t $SESSION_NAME -n 'location'
tmux send-keys -t $SESSION_NAME:1 "cd decoder_location" C-m
tmux send-keys -t $SESSION_NAME:1 "python3 train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --save_dir ./checkpoints" C-m

echo "Training started in tmux session '$SESSION_NAME'"
echo "Attach to session with: tmux attach -t $SESSION_NAME"
