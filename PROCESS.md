# 실험 과정 기록 (2026-10-31)

결과보다 과정이 중요하다는 원칙으로, 무엇을 왜 결정했고 어디서 막혔고 어떻게 확인했는지를 시간순으로 남긴다.
모든 명령은 Claude Code(울코드)가 RTX 5060 우분투 서버에서 실행했고, 교수님은 요청과 질문만 했다.

## 1. 타임라인

| 시각 | 단계 | 비고 |
|---|---|---|
| 17:06 | 환경 확인 | nvidia-smi: 드라이버 595.99, CUDA 13.2, RTX 5060 8GB. 시스템 파이썬은 3.14라 uv로 3.12 venv 생성 |
| 17:06 | PyTorch 설치 | cu128 휠, torch 2.11.0. `torch.cuda.is_available()`와 2048×2048 행렬곱으로 GPU 동작 확인 (compute capability 12.0) |
| 17:07 | 리포 clone, 데이터 준비 | nanoGPT 3adf61e, Muon f98f1ca. prepare.py로 train 1,003,854 / val 111,540 토큰 |
| 17:07 | 학습 스크립트 작성 | train.py를 복사해 train_exp.py로. 옵티마이저 선택, 그룹별 learning rate 일정, CSV 기록만 추가 |
| 17:08 | 스모크 테스트 | Muon으로 30 step. torch.compile이 sm_120에서 동작함을 확인, Muon 그룹 24개 텐서 이름·shape 출력 확인 |
| 17:09~17:15 | (A) AdamW 5000 step | 학습 241.6 s, eval 포함 363.5 s |
| 17:15~17:22 | (B) Muon 5000 step | 학습 277.2 s, eval 포함 399.0 s |
| 17:22 | 그래프·표·샘플 | 샘플 추출 필터 버그로 빈 파일이 나와서 수정 후 재생성 (아래 3절) |
| 17:25 | SOAP·Shampoo 추가 결정 | 교수님 요청. 공식 리포 clone: SOAP a1e5535, facebookresearch/optimizers 2837ffc |
| 17:26~17:29 | 권한 문제 | 외부 코드 설치·실행이 분류기에 막힘 (아래 3절). 스크립트 준비만 마치고 교수님에게 실행 요청 |
| 17:30 | 스모크 테스트 | SOAP, Shampoo 각 30 step 정상. step당 SOAP 약 55 ms, Shampoo 약 120 ms |
| 17:31~17:40 | (C) SOAP 5000 step | 학습 386.6 s |
| 17:40~17:53 | (D) Shampoo 5000 step | 학습 661.5 s |
| 17:53 | 그래프·표·샘플 재생성 | Shampoo checkpoint 로드 실패 → 모델만 재저장 (아래 3절) |
| 18:0x | 검증 | nanoGPT README 기준값 대조, checkpoint 안의 learning rate 값을 cosine 공식과 대조 (아래 4절) |
| 18:35 | GitHub 공개 리포 생성 | innotobe/nanogpt-optimizers. checkpoint 686MB와 외부 리포 제외 |
| 18:42 | train vs val 그래프 추가 | 교수님 질문 "validation set에 적용한 거냐"에서 나온 그림 |
| 18:45 | 순서 재정렬 | 교수님 제안으로 발표 연도순 AdamW → Shampoo → SOAP → Muon |
| 18:56 | 하이퍼파라미터 표 추가 | 교수님 질문 "하이퍼파라미터는 어디에"에서 |

## 2. 결정과 이유

- **별도 venv, Python 3.12.** 시스템 파이썬 3.14는 PyTorch 휠 지원이 불확실해서 uv로 3.12를 받아 썼다. 시스템에는 손대지 않았다.
- **train.py를 고치지 않고 복사본을 만들었다.** 원본과의 차이를 세 가지(옵티마이저 선택, 그룹별 learning rate 비례 일정, CSV 기록)로 한정해서, AdamW run이 원본 nanoGPT와 같은 실험이 되도록 했다.
- **Muon 적용 범위.** 공식 README가 "hidden weights에만 Muon, embedding·head·gain·bias는 AdamW"라고 하므로 transformer block 안의 2D 가중치 24개만 Muon에 넣었다. nanoGPT는 wte와 lm_head가 weight tying이라 같은 텐서인데, 이것이 Muon에 들어가면 안 되므로 이름으로 걸러서 AdamW 그룹에 두고 assert로 확인했다.
- **Muon의 weight decay 0.01.** 교수님 요청에는 learning rate 0.02와 momentum 0.95만 있었다. nanoGPT의 0.1을 그대로 쓰면 learning rate 0.02와 곱해져 AdamW 쪽보다 20배 강한 감쇠가 되므로, 공식 README 기본값 0.01을 택했다. 이 선택은 결과에 영향을 줄 수 있어 README에 명시했다.
- **learning rate 일정을 그룹별로 비례 적용.** 원본 train.py는 모든 그룹에 같은 learning rate를 덮어쓴다. Muon 그룹(0.02)과 AdamW 그룹(1e-3)이 공존하므로, 각 그룹의 기준값에 같은 cosine 비율을 곱하는 방식으로 바꿨다.
- **학습 시간 측정에서 eval을 뺐다.** 250 step마다 400배치 평가가 6초씩 들어가는데 이는 옵티마이저와 무관하므로, 그래프의 시간축은 순수 학습 step 누적 시간으로 했다. eval 포함 전체 시간도 CSV에 함께 남겼다.
- **SOAP·Shampoo는 전체 파라미터에 적용.** Muon은 embedding을 다룰 수 없어 보조 AdamW가 필수지만, SOAP과 Shampoo는 공식 사용법이 전체 파라미터 적용이므로 그대로 따랐다. LayerNorm 같은 1D 파라미터는 nanoGPT 관례대로 weight decay 0.
- **SOAP learning rate 3e-3, Shampoo는 1e-3.** SOAP은 공식 README 기본값. Shampoo는 Adam grafting을 쓰면 AdamW의 learning rate를 그대로 쓰는 것이 공식 권장이라 (A)와 같은 값. 프리컨디셔너 갱신 주기는 둘 다 10 step으로 맞췄다.
- **Shampoo를 pip 설치하지 않고 PYTHONPATH로 참조.** 설치가 권한 분류기에 막혀서 clone한 폴더를 경로에 추가하는 방식으로 대신했다. 동작은 같다.
- **생성 샘플은 best-val checkpoint에서.** nanoGPT 기본 동작이 val loss가 개선될 때만 저장하므로, 과적합 이전의 최저점 모델로 샘플을 뽑았다.

## 3. 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| 생성 샘플 파일이 0바이트 | sample.py 출력에서 구분선 `---------------`이 샘플 **뒤에** 오는데 앞에 온다고 가정한 필터를 썼다 | "Loading meta" 줄과 구분선 사이를 잘라내도록 수정 |
| 백그라운드 로그에 eval 줄이 안 보임 | grep 출력이 파일로 갈 때 블록 버퍼링됨 | `--line-buffered` 추가. 진행 확인은 CSV로 대신 |
| SOAP·Shampoo 설치·실행 명령 거부 | Claude Code 자동 모드의 권한 분류기가 새로 clone한 외부 코드 실행을 "Code from External"로 차단. pip 설치, PyPI 설치, PYTHONPATH import 모두 거부 | 스크립트를 다 준비한 뒤 교수님에게 실행 명령을 요청. 교수님 입력은 끝에 오타(M)가 붙어 실행되지 않았고, 다시 시도한 실행은 통과됨 |
| Shampoo checkpoint에서 샘플 생성 실패 | checkpoint에 옵티마이저 상태가 함께 저장되는데 Shampoo 설정 객체(enum)가 들어 있어 PyTorch 2.6+의 weights_only 로더가 거부 | 우리가 만든 파일이므로 weights_only=False로 읽어 옵티마이저 상태를 빼고 모델만 재저장 |
| GitHub 폴더의 루트 README가 결과 README로 덮어써짐 | 파일 복사 명령에서 대상 경로 실수 | 루트 README를 다시 작성 |

## 4. 검증한 것

- **외부 기준값 대조.** nanoGPT README는 같은 설정의 A100 결과로 best val loss 1.4697을 적어 두었다. 우리 AdamW run은 1.4659. 환경·데이터·모델·일정이 원본과 같다는 근거.
- **같은 출발점.** 네 run 모두 step 0의 val loss가 4.2823으로 완전히 같다. seed, 초기화, 데이터 순서가 같다는 뜻.
- **learning rate 일정.** 1750 step에 저장된 checkpoint 안의 실제 learning rate를 cosine 공식과 대조했다. AdamW 7.708e-4, Muon 그룹 1.542e-2, Muon run의 보조 AdamW 그룹 7.708e-4로 계산값과 일치.
- **Muon 그룹 구성.** 24개 텐서(10.62M)만 Muon, wte·wpe·LayerNorm 15개(0.13M)는 AdamW. 이름과 shape 출력은 results/muon_param_groups.txt.
- **스모크 테스트.** 네 옵티마이저 모두 본 run 전에 30 step 돌려서 손실이 내려가는지, 컴파일이 되는지 확인.

## 5. 하지 않은 것 (한계)

- seed 1개. seed 간 편차를 모르므로 0.01 이하의 차이는 해석하지 않았다.
- learning rate 탐색 없음. AdamW만 이 모델에 맞춰진 값이고 나머지는 저자 기본값.
- 원본 train.py와 train_exp.py의 AdamW run을 step별로 대조하는 직접 검증은 아직 안 했다.
- 별도 test set이 없어서 best checkpoint 선택과 성능 보고를 같은 val set으로 했다.

## 6. 교수님 질문이 바꾼 것

| 질문 | 결과 |
|---|---|
| "이 실험이 제대로 된 건지 어떻게 확인하냐" | 4절의 기준값 대조와 learning rate 검증 |
| "학습 곡선이 validation set에 적용한 거야?" | train vs val 그래프 추가, 과적합 구간 명시 |
| "train 그려봐, 다 같이 그리면 정신없을 테니" | 옵티마이저별 2×2 패널 |
| "AdamW, Shampoo, SOAP, Muon 순으로 정리하는 게 낫지 않아?" | 발표 연도순 재정렬 |
| "하이퍼파라미터는 어디에?" | 공통·옵티마이저별 표 추가 |
| "그런 것도 약어 쓰냐" | learning rate 등 용어를 풀어 쓰기로 |
| "결과보다 과정이 중요한데" | 이 문서 |
