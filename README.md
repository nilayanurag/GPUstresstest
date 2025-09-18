# GPU Server Quick Check

A simple setup to stress-test and validate a multi-GPU server using PyTorch.  
It downloads MNIST, trains CNN models, and can launch multiple tasks in parallel.

---

## Environment Setup

```bash
cd GPUstresstest
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

---

## Training Script

Run a single training task:

```bash
python gpu_test.py --gpu 0 --epochs 1 --batch-size 128
```

Arguments:
- `--gpu` → GPU device ID
- `--epochs` → number of training epochs
- `--batch-size` → batch size (default = 128)

---

## Batch Run (Parallel Tasks)

Run multiple tasks across GPUs in parallel:

```bash
chmod +x run_tasks.sh
./run_tasks.sh --gpus 0,1 --tasks 10 --epochs 3 --batch-size 256
```

Arguments:
- `--gpus` → GPU IDs (comma separated)
- `--tasks` → total number of tasks
- `--epochs` → epochs per task
- `--batch-size` → batch size per task
