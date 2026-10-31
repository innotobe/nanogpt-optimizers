# Evaluate next-character prediction of each best-val checkpoint on the validation set.
import os, sys, pickle, math, numpy as np, torch
sys.path.insert(0, 'nanoGPT')
from model import GPT, GPTConfig
torch.manual_seed(1337)
dev = 'cuda'
R = 'results'
runs = [('AdamW', 'adamw'), ('Shampoo', 'shampoo'), ('SOAP', 'soap'), ('Muon', 'muon')]
meta = pickle.load(open('nanoGPT/data/shakespeare_char/meta.pkl', 'rb'))
itos, stoi = meta['itos'], meta['stoi']
dec = lambda l: ''.join(itos[i] for i in l)
val = np.memmap('nanoGPT/data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')

def load(k):
    ck = torch.load(f'{R}/out-{k}/ckpt.pt', map_location=dev, weights_only=False)
    m = GPT(GPTConfig(**ck['model_args']))
    sd = {kk[len('_orig_mod.'):] if kk.startswith('_orig_mod.') else kk: v for kk, v in ck['model'].items()}
    m.load_state_dict(sd); m.to(dev).eval(); return m, ck['iter_num']

# 1) accuracy / perplexity over the whole val set, non-overlapping 256-char blocks (435 blocks)
B, T = 64, 256
nblk = (len(val) - 1) // T
X = torch.stack([torch.from_numpy(val[i*T:i*T+T].astype(np.int64)) for i in range(nblk)])
Y = torch.stack([torch.from_numpy(val[i*T+1:i*T+T+1].astype(np.int64)) for i in range(nblk)])
stats = {}
models = {}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for name, k in runs:
        m, it = load(k); models[name] = m
        nll = 0.0; top1 = 0; top5 = 0; n = 0
        for i in range(0, nblk, B):
            x, y = X[i:i+B].to(dev), Y[i:i+B].to(dev)
            logits, _ = m(x, y)  # (b, t, vocab)
            lp = torch.log_softmax(logits.float(), -1)
            nll += -lp.gather(-1, y[..., None]).sum().item()
            top = lp.topk(5, -1).indices
            top1 += (top[..., 0] == y).sum().item(); top5 += (top == y[..., None]).any(-1).sum().item()
            n += y.numel()
        stats[name] = dict(step=it, loss=nll/n, ppl=math.exp(nll/n), top1=top1/n, top5=top5/n, n=n)

# 2) continue a real val passage: 200-char prefix, greedy 120 chars, compare with ground truth
start = 50_000
prefix_ids = val[start:start+200].astype(np.int64); truth = dec(val[start+200:start+320].tolist())
cont = {}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for name, m in models.items():
        x = torch.from_numpy(prefix_ids)[None].to(dev)
        y = m.generate(x, 120, temperature=1e-6, top_k=1)[0, 200:].tolist()  # greedy
        cont[name] = dec(y)

# 3) next-char probability table at a few positions inside the prefix
pos = [60, 120, 180]
probs = {}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    x = torch.from_numpy(prefix_ids)[None].to(dev)
    for name, m in models.items():
        logits, _ = m(x, torch.from_numpy(prefix_ids)[None].to(dev))
        p = torch.softmax(logits.float(), -1)[0]
        probs[name] = {q: p[q-1].topk(3) for q in pos}

with open(f'{R}/next_char_eval.md', 'w') as f:
    f.write('# 다음 글자 예측 평가 (val set 전체 111,540 글자, best-val checkpoint)\n\n')
    f.write('| run | checkpoint step | 평균 loss | perplexity | top-1 정확도 | top-5 정확도 |\n|---|---|---|---|---|---|\n')
    for name, _ in runs:
        s = stats[name]; f.write(f"| {name} | {s['step']} | {s['loss']:.4f} | {s['ppl']:.3f} | {100*s['top1']:.2f}% | {100*s['top5']:.2f}% |\n")
    f.write('\nperplexity = exp(loss): "다음 글자 후보를 평균 몇 개 사이에서 고민하는가". top-1 = 가장 확률 높은 글자가 정답인 비율.\n')
    f.write('\n## 실제 구절 이어 쓰기 (val set 50,000번째 글자부터 200자 prefix, greedy 120자)\n\n')
    f.write('prefix (마지막 120자):\n```\n' + dec(prefix_ids[-120:].tolist()) + '\n```\n')
    f.write('정답 (실제 이어지는 120자):\n```\n' + truth + '\n```\n')
    for name, _ in runs:
        c = cont[name]; match = sum(a == b for a, b in zip(c, truth))
        f.write(f'{name} (정답과 글자 일치 {match}/120):\n```\n{c}\n```\n')
    f.write('\n## 특정 위치의 다음 글자 확률 top-3\n\n')
    for q in pos:
        ctx = dec(prefix_ids[max(0,q-30):q].tolist()).replace('\n', '⏎'); ans = itos[int(prefix_ids[q])].replace('\n', '⏎')
        f.write(f'문맥 `…{ctx}` → 정답 `{ans}`\n\n| run | 1위 | 2위 | 3위 |\n|---|---|---|---|\n')
        for name, _ in runs:
            v, i = probs[name][q]
            f.write(f'| {name} | ' + ' | '.join(f"`{itos[int(ii)].replace(chr(10),'⏎')}` {100*vv:.0f}%" for vv, ii in zip(v.tolist(), i.tolist())) + ' |\n')
        f.write('\n')
print(open(f'{R}/next_char_eval.md').read())
