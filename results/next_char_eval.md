# 다음 글자 예측 평가 (val set 전체 111,540 글자, best-val checkpoint)

| run | checkpoint step | 평균 loss | perplexity | top-1 정확도 | top-5 정확도 |
|---|---|---|---|---|---|
| AdamW | 1750 | 1.4637 | 4.322 | 57.35% | 86.00% |
| Shampoo | 1500 | 1.4725 | 4.360 | 57.21% | 85.88% |
| SOAP | 1750 | 1.4723 | 4.359 | 57.77% | 86.07% |
| Muon | 1750 | 1.4370 | 4.208 | 57.72% | 86.28% |

perplexity = exp(loss): "다음 글자 후보를 평균 몇 개 사이에서 고민하는가". top-1 = 가장 확률 높은 글자가 정답인 비율.

## 실제 구절 이어 쓰기 (val set 50,000번째 글자부터 200자 prefix, greedy 120자)

prefix (마지막 120자):
```
rous arrogance! Thou liest, thou thread,
thou thimble,
Thou yard, three-quarters, half-yard, quarter, nail!
Thou flea, t
```
정답 (실제 이어지는 120자):
```
hou nit, thou winter-cricket thou!
Braved in mine own house with a skein of thread?
Away, thou rag, thou quantity, thou 
```
AdamW (정답과 글자 일치 9/120):
```
hat the sea which was the world with the world,
And the sea of the sea-shepherd's son,
And the sea-shepherd's son, and t
```
Shampoo (정답과 글자 일치 7/120):
```
he season will be the seas of thy heart.

PETRUCHIO:
A gentleman, sir, a man that thou hast stay'd
to thee and the world
```
SOAP (정답과 글자 일치 9/120):
```
hat thou hast straight to the field
Of thy breast father's son: thou art a man to be so barren
As thou art as thou as th
```
Muon (정답과 글자 일치 17/120):
```
hat thou hast not spoke to the prince,
That thou hast seen the seas of the seas,
And that thou hast spoke the seas of th
```

## 특정 위치의 다음 글자 확률 top-3

문맥 `…means to make⏎a puppet of her.` → 정답 `⏎`

| run | 1위 | 2위 | 3위 |
|---|---|---|---|
| AdamW | `⏎` 59% | ` ` 41% | `'` 0% |
| Shampoo | `⏎` 79% | ` ` 20% | `'` 1% |
| SOAP | `⏎` 76% | ` ` 23% | `'` 0% |
| Muon | `⏎` 59% | ` ` 41% | `'` 0% |

문맥 `…ance! Thou liest, thou thread,` → 정답 `⏎`

| run | 1위 | 2위 | 3위 |
|---|---|---|---|
| AdamW | ` ` 50% | `⏎` 50% | `-` 0% |
| Shampoo | `⏎` 93% | ` ` 7% | `-` 0% |
| SOAP | ` ` 79% | `⏎` 20% | `-` 1% |
| Muon | ` ` 54% | `⏎` 45% | `-` 1% |

문맥 `…e-quarters, half-yard, quarter` → 정답 `,`

| run | 1위 | 2위 | 3위 |
|---|---|---|---|
| AdamW | `s` 34% | ` ` 29% | `'` 11% |
| Shampoo | `'` 28% | ` ` 28% | `s` 21% |
| SOAP | `'` 27% | ` ` 24% | `,` 21% |
| Muon | `,` 27% | `'` 14% | ` ` 13% |

