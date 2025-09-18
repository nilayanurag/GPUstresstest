#!/bin/bash
# setup.sh
# Usage: ./setup.sh

set -e  # Exit on error

# ---- Config ----
PYTHON_BIN=python3
VENV_DIR=.venv

# ---- Step 1: Create virtual environment ----
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment in $VENV_DIR"
  $PYTHON_BIN -m venv $VENV_DIR
else
  echo "[INFO] Virtual environment already exists at $VENV_DIR"
fi

# ---- Step 2: Activate venv ----
echo "[INFO] Activating virtual environment"
source $VENV_DIR/bin/activate

# ---- Step 3: Upgrade pip ----
echo "[INFO] Upgrading pip"
pip install --upgrade pip

# ---- Step 4: Install all requirements ----
if [ -f requirements.txt ]; then
  echo "[INFO] Installing requirements.txt"
  pip install -r requirements.txt
else
  echo "[WARN] No requirements.txt found. Skipping."
fi

echo "[INFO] Setup complete. To activate run: source $VENV_DIR/bin/activate"