# QUICKSTART — 클라우드 노트북 pod 실행

RunPod 같은 클라우드 노트북 pod 에 접속한 뒤 **복사해서 붙이면 되는 명령만** 모았습니다.
설명은 [`00_README.md`](00_README.md), 막히면 [`04_TROUBLESHOOTING.md`](04_TROUBLESHOOTING.md).

> **GPU 는 필요하지 않습니다.** 이 파이프라인은 LightGBM(CPU 전용)이 주력이고
> TensorFlow 모델이 매우 작아 GPU 이득이 없습니다. GPU pod 에서도 정상 동작하지만
> **CPU pod 이 더 저렴합니다.**

---

## 0. pod 고르기

| 항목 | 권장 |
|---|---|
| 이미지 | Python 3.10~3.12 가 있는 아무 이미지 (PyTorch/TensorFlow 계열 무엇이든) |
| GPU | **불필요.** CPU-only 인스턴스로 충분 |
| vCPU | 4코어 이상 권장 (LightGBM `num_threads=4` 고정) |
| RAM | 4GB 이상 (데이터가 428KB 라 여유롭다) |
| 디스크 | 5GB 이상 (TensorFlow wheel 이 큼) |

검증 환경은 **Python 3.11.9** 입니다. 3.10~3.12 를 허용합니다
(`pandas 3.0` 은 ≥3.10, `tensorflow 2.21` 은 ≤3.12).

---

## 1. 파일 올리기

pod 터미널에서:

```bash
cd /workspace          # RunPod 은 보통 /workspace 가 영구 볼륨
# (이 폴더를 압축해 업로드했다면)
unzip KAMP_공모전.zip
cd KAMP_공모전
ls data/okm_augumented_2021.csv   # 이 파일이 보여야 한다
```

> ⚠️ **영구 볼륨에 두세요.** pod 을 재시작하면 볼륨 밖의 파일은 사라집니다.
> RunPod 은 `/workspace`, 다른 서비스는 `/persist`·`/data` 등 이름이 다를 수 있습니다.

---

## 2. 셋업 (3~5분)

```bash
bash setup_pod.sh
```

이 스크립트가 하는 일:

1. **한글 폰트 설치** (`fonts-nanum`) + matplotlib 폰트 캐시 초기화
   — 이걸 안 하면 그림의 한글이 `□□□` 로 깨집니다
2. **`numpy` · `pandas` 를 먼저 고정** — pod 이미지에 선설치된 pandas 2.x 를 3.0.5 로 덮어씁니다
3. 나머지 런타임 패키지 설치
4. Jupyter 커널 `KAMP 자원최적화` 등록
5. 버전 대조 — 하나라도 어긋나면 **0이 아닌 코드로 종료**

여러 번 실행해도 안전합니다(idempotent).

테스트·재빌드까지 하려면:

```bash
KAMP_DEV=1 bash setup_pod.sh
```

---

## 3. 환경 점검 (30초) — 건너뛰지 마세요

Jupyter UI 에서 **`START_HERE.ipynb`** 를 열고 커널 `KAMP 자원최적화` 선택 → `Run All`.

또는 터미널에서:

```bash
bash run.sh check      # make 가 있으면 make check
```

점검 항목: 작업 디렉터리 · 패키지 버전 · **한글 실제 렌더** · 디스크 · GPU 인식.
문제가 있으면 마지막 셀에서 **무엇을 해야 하는지** 알려주고 멈춥니다.

---

## 4. 실행

### A) 배치 실행 — **권장**

FULL 실행은 CPU 에서 **30~60분**입니다. Jupyter UI 로 돌리다 브라우저·세션이 끊기면
처음부터 다시 돌려야 합니다. 배치로 돌리면 세션과 무관하게 끝까지 갑니다.

```bash
bash run.sh run        # make 가 있으면 make run
```

또는 직접:

```bash
mkdir -p outputs
nohup env PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8 \
  python -X utf8 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  자원최적화_제안모델.ipynb > outputs/run.log 2>&1 &

# 진행 확인
tail -f outputs/run.log
```

`nohup ... &` 로 띄우면 터미널을 닫아도 계속 돕니다.

### B) Jupyter UI

`자원최적화_제안모델.ipynb` 를 열고 `Kernel → Restart Kernel and Run All Cells`.
셀을 위에서 아래로 읽으면 무엇을 계산했고 그 결과가 어떻게 생겼는지 이어집니다.

### C) 빠른 확인만 (3~5분)

구조가 도는지만 보고 싶을 때. **수치는 최종값이 아닙니다**
(Optuna 100회→6회, 에폭 500→5로 축소).

```bash
bash run.sh run-fast   # make 가 있으면 make run-fast
```

---

## 5. 실행 후 — 반드시

```bash
bash run.sh finalize   # make 가 있으면 make finalize
```

노트북 셀 출력에 남은 **stderr 경고를 제거하고 개인정보를 후스캔**합니다.

왜 별도 단계인가: `nbconvert --inplace` 는 모든 셀이 끝난 **뒤에** 파일을 씁니다.
그래서 노트북 안(10.5절)의 개인정보 스캔은 자기 실행 출력을 볼 수 없습니다.
서드파티 경고가 stderr 로 내보내는 메시지에 **설치 경로(사용자명 포함)** 가 박히는데,
블라인드 평가에서는 위반입니다. 실제로 이 경로로 6건이 남은 것을 잡았습니다.

---

## 6. 결과 보기

```bash
ls outputs/figures/ | head            # 그림 39장
ls outputs/tables/ | wc -l            # 표 108개
cat outputs/report_tbd_filled.md      # 보고서 채움표 + 문장 치환사전
head -3 outputs/predictions_test_336h.csv
```

| 파일 | 내용 |
|---|---|
| `outputs/report_tbd_filled.md` | 보고서 빈칸에 넣을 값 전부 + 고쳐야 할 문장 + 장별 PASS/FAIL |
| `outputs/figures/figure_index.csv` | 그림 ↔ 보고서 절 매핑 |
| `outputs/predictions_test_336h.csv` | 제출용 예측결과 336행 |
| `outputs/report_ch4_draft.md` 외 | 보고서 4·5·6장 본문 초안 |

---

## 7. 검증 (선택)

```bash
bash run.sh test      # 검증 테스트 236개 (약 4분)
bash run.sh verify    # 동일 시드 2회 실행 비교 — 722개 셀 일치 (약 8분)
bash run.sh build     # src/ 에서 노트북 재조립 → 159셀
```

테스트에는 **누수 탐지기 자체를 검증하는 음성 대조**가 들어 있습니다.
일부러 누수 피처(`rolling().shift(1)` 등)를 주입하면 탐지기가 반드시 실패하는지 확인합니다.

---

## 명령 요약

```bash
bash setup_pod.sh          # 1회 셋업
bash run.sh check          # 환경 점검
bash run.sh run            # 본실행 (30~60분)
bash run.sh finalize       # 실행 후 필수
bash run.sh test           # 검증 테스트
bash run.sh verify         # 재현성 검사
```

`make` 가 있으면 `make check` / `make run` / `make clean` 등도 쓸 수 있습니다
(`make help` 로 목록 확인).
