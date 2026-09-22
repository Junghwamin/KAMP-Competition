#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 클라우드 노트북 pod(RunPod 등) 원샷 셋업
#
#   bash setup_pod.sh
#
# 여러 번 실행해도 안전하다(idempotent). 한 단계라도 실패하면 즉시 멈춘다.
#
# ⚠️ 이 스크립트는 Linux(Debian/Ubuntu 계열) pod 기준이다.
#    Windows/macOS 로컬에서는 `pip install -r requirements.txt` 만 하면 된다.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

say()  { printf '\n\033[1m── %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")"
ROOT="$(pwd)"

# ── 0. 위치 확인 ─────────────────────────────────────────────────────────────
# 노트북이 CWD 기준 상대경로(`data/...`, `outputs/...`)를 쓰므로
# 반드시 이 폴더를 작업 디렉터리로 삼아야 한다.
say "0. 패키지 위치 확인"
[ -f "data/okm_augumented_2021.csv" ] \
  || die "data/okm_augumented_2021.csv 가 없다. 이 스크립트를 패키지 루트에서 실행했는지 확인하라."
ok "패키지 루트: $ROOT"

# ── 1. 한글 폰트 ─────────────────────────────────────────────────────────────
# 코드의 폰트 폴백 체인은
#   ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic")
# 이다. Malgun Gothic 은 Windows 전용이므로 Linux 에서는 NanumGothic 을 깔면
# **코드 수정 없이** 자동으로 잡힌다. 폰트가 하나도 없으면 그림의 한글이 □□□ 로 깨진다.
say "1. 한글 폰트 (matplotlib 그림용)"
if fc-list 2>/dev/null | grep -qi nanum; then
  ok "NanumGothic 이미 설치됨"
else
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq
    # fonts-nanum: 본문용,  fonts-nanum-coding: 고정폭(선택)
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-nanum fonts-nanum-coding
    fc-cache -f >/dev/null 2>&1 || true
    ok "fonts-nanum 설치 완료"
  else
    warn "root 가 아니라 apt 설치를 건너뛴다."
    warn "그림의 한글이 깨진다. 수동 설치: sudo apt-get install -y fonts-nanum"
  fi
fi

# matplotlib 은 폰트 목록을 캐시한다. 새로 깐 폰트를 인식시키려면
# **캐시를 반드시 지워야** 한다. 이 한 줄이 없으면 설치해도 한글이 깨진다.
rm -rf "${HOME}/.cache/matplotlib" 2>/dev/null || true
ok "matplotlib 폰트 캐시 초기화"

# ── 2. numpy·pandas 를 먼저 고정 ─────────────────────────────────────────────
# pod 이미지에는 보통 pandas 2.x 가 선설치되어 있다. 이 프로젝트는 3.0.5 에서
# 검증되었고, 무음실패 탐지(ChainedAssignmentError 승격)가 3.x 동작에 의존한다.
# 다른 패키지보다 **먼저** 올려야 의존 해결이 꼬이지 않는다.
say "2. numpy · pandas 고정 설치 (pod 선설치 버전을 덮어씀)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet --no-cache-dir "numpy==2.3.5" "pandas==3.0.5"
ok "numpy 2.3.5 · pandas 3.0.5"

# ── 3. 나머지 런타임 ─────────────────────────────────────────────────────────
say "3. 런타임 패키지 설치"
python -m pip install --quiet --no-cache-dir -r requirements.txt
ok "requirements.txt 설치 완료"

if [ "${KAMP_DEV:-0}" = "1" ]; then
  python -m pip install --quiet --no-cache-dir -r requirements-dev.txt
  ok "requirements-dev.txt 설치 완료 (재빌드·테스트용)"
fi

# ── 4. Jupyter 커널 등록 ─────────────────────────────────────────────────────
say "4. Jupyter 커널 등록"
python -m ipykernel install --user --name kamp --display-name "KAMP 자원최적화 (Python 3.11)" >/dev/null 2>&1
ok "커널 'KAMP 자원최적화' 등록 — Jupyter 에서 이 커널을 선택하라"

# ── 5. 설치 검증 ─────────────────────────────────────────────────────────────
# 버전이 하나라도 어긋나면 0 이 아닌 코드로 종료해 사용자가 바로 알게 한다.
say "5. 설치 검증"
python - <<'PY'
import sys, importlib
EXPECT = {
    "numpy": "2.3.5", "pandas": "3.0.5", "sklearn": "1.9.1",
    "scipy": "1.17.1", "matplotlib": "3.11.1", "lightgbm": "4.7.0",
    "optuna": "5.0.0", "shap": "0.51.0", "statsmodels": "0.14.6",
}
bad = []
for mod, want in EXPECT.items():
    try:
        got = importlib.import_module(mod).__version__
    except Exception as e:
        bad.append(f"{mod}: import 실패 ({type(e).__name__})"); continue
    mark = "OK " if got == want else "!! "
    print(f"   {mark}{mod:<14s} {got}" + ("" if got == want else f"  (기대 {want})"))
    if got != want:
        bad.append(f"{mod}: {got} != {want}")

try:
    import tensorflow as tf
    print(f"   OK {'tensorflow':<14s} {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"      GPU 인식: {len(gpus)}개 "
          f"({'사용 가능하지만 이 파이프라인은 GPU를 쓰지 않는다' if gpus else 'CPU 모드 — 정상'})")
except Exception as e:
    print(f"   !  tensorflow 미설치/로드실패 ({type(e).__name__}) "
          f"— SimpleRNN·DNN 만 건너뛰고 나머지는 정상 실행된다")

py = ".".join(map(str, sys.version_info[:3]))
print(f"   {'OK ' if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else '!! '}"
      f"{'python':<14s} {py}  (권장 3.11, 허용 3.10~3.12)")

if bad:
    print("\n버전 불일치:", *bad, sep="\n  - ")
    sys.exit(1)
PY

say "완료"
cat <<'MSG'
   다음 단계:
     1) Jupyter 에서 START_HERE.ipynb 를 열고 커널 'KAMP 자원최적화' 선택 → 전체 실행
     2) 또는 배치로:   make run        (로그: outputs/run.log)

   ⚠️ 이 파이프라인은 GPU 를 쓰지 않는다 (LightGBM = CPU 전용, TF 모델이 매우 작음).
      GPU pod 에서도 정상 동작하지만 CPU pod 이 더 저렴하다.
MSG
