#!/bin/bash
# SOAP + Shampoo runs, same settings as run_all.sh. Usage: ./run_extra.sh [soap shampoo]
set -e
cd ~/research/nanogpt-muon/nanoGPT
PY=../.venv/bin/python
export PYTHONPATH=../fb_optimizers:$PYTHONPATH
RUNS=${@:-soap shampoo}
SMOKE=../logs/smoke
mkdir -p $SMOKE
for r in $RUNS; do   # quick 30-iter smoke test first so errors show up in seconds
  echo "===== SMOKE $r $(date) ====="
  $PY train_exp.py config/train_shakespeare_char.py --optimizer_name=$r --max_iters=30 --eval_interval=10 --eval_iters=5 \
      --out_dir=$SMOKE/out-$r --csv_path=$SMOKE/$r.csv 2>&1 | grep --line-buffered -E "^step|^===|Error|Traceback|^  "
done
for r in $RUNS; do
  echo "===== RUN $r start $(date) ====="
  $PY train_exp.py config/train_shakespeare_char.py --optimizer_name=$r \
      --out_dir=../results/out-$r --csv_path=../results/log_$r.csv 2>&1 | tee ../logs/train_$r.log | grep --line-buffered -v "^iter "
  echo "===== RUN $r end $(date) ====="
done
echo ALL_DONE
