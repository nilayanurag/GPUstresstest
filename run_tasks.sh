#!/bin/bash
# Usage:
#   ./run_tasks.sh --gpus 0,1 --tasks 10 --epochs 3 --batch-size 128

# ---- Default values ----
BATCH_SIZE=128

# ---- Parse arguments ----
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gpus) GPU_LIST="$2"; shift ;;
    --tasks) NUM_TASKS="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    -h|--help)
      echo "Usage: $0 --gpus <ids> --tasks <num_tasks> --epochs <num_epochs> [--batch-size N]"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done

# ---- Validate ----
if [[ -z "$GPU_LIST" || -z "$NUM_TASKS" || -z "$EPOCHS" ]]; then
  echo "[ERROR] Missing required arguments."
  echo "Usage: $0 --gpus <ids> --tasks <num_tasks> --epochs <num_epochs> [--batch-size N]"
  exit 1
fi

# ---- Setup ----
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
mkdir -p logs

echo "[INFO] GPUs to use: ${GPUS[*]}"
echo "[INFO] Total tasks: $NUM_TASKS"
echo "[INFO] Epochs per task: $EPOCHS"
echo "[INFO] Batch size: $BATCH_SIZE"
echo "[INFO] Tasks per GPU (approx): $((NUM_TASKS / NUM_GPUS))"

# ---- Launch tasks ----
for ((i=0; i<NUM_TASKS; i++)); do
  GPU_INDEX=$((i % NUM_GPUS))
  GPU_ID=${GPUS[$GPU_INDEX]}
  echo "[LAUNCH] Task $i on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID python gpu_test.py \
      --gpu 0 --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" > logs/task_$i.log 2>&1 &
done

echo "[INFO] All tasks launched. Logs are in ./logs/"
wait
echo "[INFO] All tasks finished."
