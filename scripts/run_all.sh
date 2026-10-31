#!/bin/bash
set -e
cd ~/research/nanogpt-muon/nanoGPT
PY=../.venv/bin/python
for cfg in "adamw adamw" "muon muon"; do
  set -- $cfg
  echo "===== RUN $1 start $(date) ====="
  $PY train_exp.py config/train_shakespeare_char.py --optimizer_name=$1 \
      --out_dir=../results/out-$1 --csv_path=../results/log_$1.csv 2>&1 | tee ../logs/train_$1.log | grep -v "^iter "
  echo "===== RUN $1 end $(date) ====="
done
echo ALL_DONE
