#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! conda run -n squidiff python --version >/dev/null 2>&1; then
  conda env create -f environment-portable.yml -y
else
  conda install -n squidiff --freeze-installed -c conda-forge git -y
fi
conda run --no-capture-output -n squidiff python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n squidiff python -m pip install -r requirements.txt
conda run --no-capture-output -n squidiff python -m pip install -e upstream/Squidiff --no-deps
conda run --no-capture-output -n squidiff python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
