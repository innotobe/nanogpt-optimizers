# nanoGPT shakespeare-char: AdamW vs Shampoo vs SOAP vs Muon (RTX 5060, 2026-10-31)

옵티마이저는 발표 순서대로 정리한다: AdamW(2017) → Shampoo(2018) → SOAP(2024.9) → Muon(2024.10).

공통 설정: n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, dropout=0.2,
max_iters=5000, warmup 100 + cosine decay(min_lr = lr/10), seed=1337, eval_interval=250, eval_iters=200, bf16, torch.compile.
네 run 모두 같은 warmup/cosine 스케줄(각 옵티마이저의 base lr에 비례).

- (A) AdamW: lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1 (2D), 0 (1D)
- (B) Shampoo (facebookresearch/optimizers DistributedShampoo, 전체 파라미터): AdamW grafting, (A)와 같은 lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1(2D) / 0(1D), epsilon=1e-12, grafting epsilon=1e-8, max_preconditioner_dim=1536, precondition_frequency=10.
- (C) SOAP (nikhilvyas/SOAP 공식 soap.py, 전체 파라미터): lr=3e-3, betas=(0.95, 0.95), weight_decay=0.01(2D) / 0(1D), precondition_frequency=10 (공식 README 기본값). 1D 파라미터는 프리컨디셔닝 없이 Adam.
- (D) Muon + aux AdamW (KellerJordan/Muon, SingleDeviceMuonWithAuxAdam):
  - Muon: transformer block 내부 2D 가중치 24개(c_attn, c_proj, c_fc, c_proj) 10.62M 파라미터, lr=0.02, momentum=0.95, weight_decay=0.01(공식 README 기본값)
  - aux AdamW: wte(lm_head와 tied), wpe, LayerNorm 15개 0.13M 파라미터, (A)와 같은 lr=1e-3, betas, weight_decay
  - 그룹 구성: muon_param_groups.txt

## 결과 표
| run | final val loss | best val loss (step) | final train loss | train time (s) | wall time incl. eval (s) | ms/step |
|---|---|---|---|---|---|---|
| AdamW (2017) | 1.7005 | 1.4659 (1750) | 0.6183 | 241.6 | 363.5 | 48.5 |
| Shampoo (2018, Adam graft) | 1.8481 | 1.4745 (1500) | 0.4341 | 661.5 | 784.1 | 132.8 |
| SOAP (2024) | 1.7869 | 1.4746 (1750) | 0.4912 | 386.6 | 514.3 | 77.6 |
| Muon (2024, +aux AdamW) | 1.6071 | **1.4385** (1750) | 0.6538 | 277.2 | 399.0 | 55.7 |

| val loss threshold | AdamW: step (time s) | Shampoo: step (time s) | SOAP: step (time s) | Muon: step (time s) |
|---|---|---|---|---|
| ≤ 2.0 | 500 (24s) | 250 (33s) | 250 (19s) | 250 (14s) |
| ≤ 1.8 | 500 (24s) | 500 (67s) | 500 (39s) | 250 (14s) |
| ≤ 1.7 | 750 (36s) | 500 (67s) | 500 (39s) | 500 (28s) |
| ≤ 1.6 | 750 (36s) | 750 (100s) | 750 (58s) | 500 (28s) |
| ≤ 1.5 | 1250 (60s) | 1000 (133s) | 1250 (97s) | 1000 (55s) |

train time = eval 시간을 뺀 순수 학습 step 누적 시간, wall time = eval 포함 전체 경과 시간.
nanoGPT README의 A100 기준값은 best val 1.4697 (우리 AdamW 1.4659와 일치).

## 파일
- log_{adamw,shampoo,soap,muon}.csv: step, train loss, val loss, 누적 학습 시간, 누적 전체 시간
- val_loss_vs_step.png, val_loss_vs_time.png (4개 run), train_val_curves.png (옵티마이저별 train vs val)
- sample_*.txt: best-val checkpoint에서 500자 생성(temperature 0.8, top_k 200, seed 1337)
- out-*/: best-val checkpoint (out-shampoo는 옵티마이저 상태를 제거하고 모델만 저장)

## 요약
- step 기준 최저 val loss: Muon 1.439 < AdamW 1.466 ≈ Shampoo 1.4745 ≈ SOAP 1.4746. Muon만 AdamW보다 확실히 좋고, Shampoo·SOAP은 초반(250~1000 step)에는 AdamW보다 빠르지만 최저점은 AdamW와 같은 수준이다.
- 시간 기준에서는 순위가 바뀐다. step당 시간이 AdamW 48 ms, Shampoo 133 ms, SOAP 78 ms, Muon 56 ms라서, 같은 시간 안에 도달하는 val loss는 Muon > AdamW > SOAP > Shampoo 순이다. Shampoo는 step 효율 이득이 2.7배 느린 step 비용을 상쇄하지 못한다.
- 네 run 모두 1500~1750 step 이후 과적합. Shampoo·SOAP은 train loss를 더 낮게 밀어붙이고(0.43~0.49) 그만큼 val loss가 더 빨리 올라간다. 이 데이터셋(1M 토큰)에서는 2차 옵티마이저의 강한 피팅력이 일반화에 불리하게 작용했다.
- 단일 seed, lr 탐색 없음. SOAP은 공식 기본 lr(3e-3)을, Shampoo는 grafting 덕에 AdamW lr을 그대로 썼으나 둘 다 이 모델 크기에 맞춰 튜닝된 값은 아니다.
