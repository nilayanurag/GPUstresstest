#!/bin/bash
# Usage: ./run_tasks.sh "0,1" 10 3
# -> runs 10 tasks split across GPU 0 and 1, each for 3 epochs

if [ $# -lt 3 ]; then
  echo "Usage: $0 <gpu_ids_comma_sep> <num_tasks> <epochs>"
  exit 1
fi

GPU_LIST=$1
NUM_TASKS=$2
EPOCHS=$3

# Convert comma-separated string to array
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

mkdir -p logs

echo "[INFO] GPUs to use: ${GPUS[*]}"
echo "[INFO] Total tasks: $NUM_TASKS"
echo "[INFO] Epochs per task: $EPOCHS"
echo "[INFO] Tasks per GPU (approx): $((NUM_TASKS / NUM_GPUS))"

for ((i=0; i<NUM_TASKS; i++)); do
  GPU_INDEX=$((i % NUM_GPUS))
  GPU_ID=${GPUS[$GPU_INDEX]}
  echo "[LAUNCH] Task $i on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID python gpu_test.py \
      --gpu 0 --epochs "$EPOCHS" --batch-size 128 > logs/task_$i.log 2>&1 &
done

echo "[INFO] All tasks launched. Logs are in ./logs/"
wait
echo "[INFO] All tasks finished."
