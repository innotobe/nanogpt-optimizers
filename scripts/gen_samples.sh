#!/bin/bash
cd ~/research/nanogpt-muon/nanoGPT
for r in adamw shampoo soap muon; do
  [ -f ../results/out-$r/ckpt.pt ] || continue
  ../.venv/bin/python sample.py --out_dir=../results/out-$r --num_samples=1 --max_new_tokens=500 --seed=1337 2>/dev/null \
    | sed -n '/^Loading meta/,/^---------------$/p' | sed '1d;$d' > ../results/sample_$r.txt
  echo "== $r: $(wc -c < ../results/sample_$r.txt) chars"
done
