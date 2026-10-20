#!/usr/bin/env bash
set -euo pipefail
nvidia-smi
ncu --version
python3 -u verify.py
python3 -u measure.py --phase eager
ncu --profile-from-start off --target-processes all \
  --metrics gpu__time_duration.sum --clock-control none --cache-control all \
  --kernel-name-base demangled --force-overwrite -o comparison \
  python3 -u measure.py --phase ncu 2>&1 | tee ncu.log
ncu --import comparison.ncu-rep --csv --page raw --log-file metrics.csv
python3 export_results.py
