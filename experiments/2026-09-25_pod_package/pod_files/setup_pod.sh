#!/usr/bin/env bash
# 포드 1회 셋업 — 공유 conda 환경 · 비root 를 기본으로 가정한다.
#
#   bash setup_pod.sh              # 없는 패키지만 설치 (버전 핀 안 함)
#   bash setup_pod.sh --pin        # requirements.txt 의 버전으로 정확히 맞춤 (단독 환경일 때만)
#   bash setup_pod.sh --venv       # ~/.venvs/kamp-exp 가상환경을 만들어 거기에 설치
#
# 설계 원칙
#   1. 남의 환경을 건드리지 않는다 — base 에 쓰기 권한이 없으면 --user 나 venv 로 간다.
#   2. **이미 있는 건 건드리지 않는다** — import 되는 패키지는 재설치·업그레이드하지 않는다.
#   3. 버전을 억지로 맞추지 않는다 — 공유 환경에서 핀 강제는 남의 코드를 깨뜨린다.
set -uo pipefail
cd "$(dirname "$0")"

PIN=0; FORCE_VENV=0
for a in "$@"; do
    case "$a" in
        --pin)  PIN=1 ;;
        --venv) FORCE_VENV=1 ;;
        *) echo "모르는 옵션: $a"; exit 2 ;;
    esac
done

ok()   { printf '  \033[32m✔\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✘\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }

PY=python

step 1/5 "설치 위치 정하기"
WRITABLE=$($PY -c "import os,sysconfig;print(1 if os.access(sysconfig.get_paths()['purelib'],os.W_OK) else 0)")
USERSITE=$($PY -c "import site;print(1 if site.ENABLE_USER_SITE else 0)")
PIP_TARGET=""
VENV_DIR="$HOME/.venvs/kamp-exp"

if [ "$FORCE_VENV" = "1" ] || { [ "$WRITABLE" = "0" ] && [ "$USERSITE" = "0" ]; }; then
    # 공유 환경의 큰 패키지(TF 등)를 그대로 재사용하고 부족한 것만 venv 에 얹는다
    if [ ! -d "$VENV_DIR" ]; then
        $PY -m venv --system-site-packages "$VENV_DIR" || { bad "venv 생성 실패"; exit 1; }
    fi
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    PY=python
    ok "가상환경 사용: $VENV_DIR (base 패키지 상속)"
    echo "     주피터에서는 커널 'KAMP 자원최적화 (실험전용)' 을 골라야 한다."
elif [ "$WRITABLE" = "1" ]; then
    ok "현재 환경에 직접 설치 (쓰기 권한 있음)"
else
    PIP_TARGET="--user"
    ok "사용자 영역에 설치 (--user) — base 환경은 건드리지 않는다"
fi

step 2/5 "이미 있는 패키지 확인"
MISSING=$($PY - <<'PY'
import importlib.util
# pip 이름:임포트 이름
need = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "scikit-learn": "sklearn", "matplotlib": "matplotlib",
        "lightgbm": "lightgbm", "optuna": "optuna",
        "statsmodels": "statsmodels", "tensorflow": "tensorflow", "shap": "shap",
        "ipykernel": "ipykernel"}
missing = [p for p, m in need.items() if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)
$PY - <<'PY'
import importlib.metadata as md, importlib.util
need = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "scikit-learn": "sklearn", "matplotlib": "matplotlib",
        "lightgbm": "lightgbm", "optuna": "optuna",
        "statsmodels": "statsmodels", "tensorflow": "tensorflow", "shap": "shap",
        "ipykernel": "ipykernel"}
for p, m in need.items():
    if importlib.util.find_spec(m) is None:
        print(f"  ✘ {p:14s} 없음 — 설치 대상")
    else:
        try:
            v = md.version(p)
        except Exception:
            v = "?"
        print(f"  ✔ {p:14s} {v} (그대로 둔다)")
PY

step 3/5 "없는 것만 설치"
# 이미 깔린 것들을 constraints 로 묶는다. 이렇게 해야 새 패키지의 의존성 해석이
# numpy·TF 를 조용히 갈아끼우지 못하고, 충돌하면 '설치 실패'로 드러난다.
$PY - > /tmp/kamp_constraints.txt <<'PY'
import importlib.metadata as md
for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "lightgbm",
          "statsmodels", "tensorflow", "keras", "numba", "llvmlite"]:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception:
        pass
PY
echo "  기존 버전 고정(constraints):"
sed 's/^/     /' /tmp/kamp_constraints.txt

if [ -z "${MISSING// /}" ]; then
    ok "설치할 게 없다"
elif [ "$PIN" = "1" ]; then
    warn "--pin: requirements.txt 의 버전으로 맞춘다 (공유 환경이면 위험)"
    $PY -m pip install --quiet --no-cache-dir $PIP_TARGET -r requirements.txt \
        && ok "완료" || { bad "설치 실패 — 아래 '막혔을 때' 를 볼 것"; }
else
    echo "  설치 대상: $MISSING"
    # shellcheck disable=SC2086
    $PY -m pip install --quiet --no-cache-dir $PIP_TARGET \
        -c /tmp/kamp_constraints.txt $MISSING \
        && ok "완료 (기존 패키지는 그대로)" \
        || { bad "설치 실패 — 기존 버전과 충돌했을 수 있다. 아래 '막혔을 때' 참고"; }
fi

# pip 가 교체 도중 죽으면 site-packages 에 '~xxx' 잔해가 남아 경고를 뿌린다
LEFT=$($PY -c "
import sysconfig, pathlib
sp = pathlib.Path(sysconfig.get_paths()['purelib'])
print(' '.join(p.name for p in sp.glob('~*')))" 2>/dev/null)
if [ -n "${LEFT// /}" ]; then
    warn "이전 설치 실패의 잔해가 있다: $LEFT"
    echo "     정상 동작하는지 확인 후 지우면 된다:"
    echo "       python -c 'import numpy, pandas; print(numpy.__version__, pandas.__version__)'"
    for d in $LEFT; do
        echo "       rm -rf $($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")/$d"
    done
fi

step 4/5 "한글 폰트"
FOUND=$($PY - <<'PY'
from matplotlib import font_manager
names = {f.name for f in font_manager.fontManager.ttflist}
hit = [n for n in names if any(k in n.lower() for k in
       ("nanum", "noto sans cjk", "noto sans kr", "malgun", "gothic", "gulim", "batang"))]
print(hit[0] if hit else "")
PY
)
if [ -n "$FOUND" ]; then
    ok "한글 폰트 있음: $FOUND"
elif fc-list :lang=ko 2>/dev/null | grep -q .; then
    warn "시스템에는 한국어 폰트가 있는데 matplotlib 이 아직 모른다 → 캐시를 지운다"
    rm -rf "$HOME/.cache/matplotlib" 2>/dev/null || true
    ok "matplotlib 폰트 캐시 삭제 — 커널을 다시 시작하면 잡힌다"
elif [ "$(id -u)" = "0" ]; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-nanum \
        && fc-cache -f >/dev/null 2>&1 && rm -rf "$HOME/.cache/matplotlib" && ok "fonts-nanum 설치"
else
    warn "root 가 아니고 한글 폰트도 없다 → 홈에 직접 내려받기를 시도한다"
    mkdir -p "$HOME/.fonts"
    URL="https://github.com/googlefonts/nanumgothic/raw/main/fonts/ttf/NanumGothic-Regular.ttf"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "$HOME/.fonts/NanumGothic.ttf" 2>/dev/null
    else
        wget -q "$URL" -O "$HOME/.fonts/NanumGothic.ttf" 2>/dev/null
    fi
    # 받은 게 진짜 폰트인지 확인한다 — 404 HTML 을 폰트로 착각하면 조용히 깨진다
    if $PY - <<'PY'
import sys, pathlib
from matplotlib import font_manager
p = pathlib.Path.home() / ".fonts" / "NanumGothic.ttf"
try:
    if p.stat().st_size < 50_000:
        raise ValueError(f"너무 작다({p.stat().st_size}B) — 폰트가 아니다")
    name = font_manager.FontProperties(fname=str(p)).get_name()
    print(f"  받은 폰트 이름: {name}")
    sys.exit(0)
except Exception as e:
    print(f"  실패: {e}")
    sys.exit(1)
PY
    then
        fc-cache -f "$HOME/.fonts" >/dev/null 2>&1 || true
        rm -rf "$HOME/.cache/matplotlib" 2>/dev/null || true
        ok "홈에 폰트 설치 — 커널을 다시 시작할 것"
    else
        rm -f "$HOME/.fonts/NanumGothic.ttf"
        bad "폰트를 못 구했다. 그림의 한글이 □□□ 로 깨진다(코드는 죽지 않는다)."
        echo "     해결: 관리자에게 'apt-get install -y fonts-nanum' 요청,"
        echo "     또는 NanumGothic.ttf 를 직접 ~/.fonts/ 에 올린 뒤 rm -rf ~/.cache/matplotlib"
    fi
fi

step 5/5 "최종 점검"
$PY - <<'PY'
import importlib, sys
from pathlib import Path
need = ["numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "lightgbm", "optuna", "statsmodels", "tensorflow", "shap"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  ✔ {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append(m); print(f"  ✘ {m:12s} {type(e).__name__}: {e}")
try:
    import tensorflow as tf
    g = tf.config.list_physical_devices("GPU")
    print(f"\n  GPU: {g if g else '없음 (CPU 로 실행 — 결과는 같고 DNN 부분만 느려진다)'}")
except Exception as e:
    print(f"\n  GPU 확인 불가: {type(e).__name__}")
csv = Path("data/okm_augumented_2021.csv")
print(f"  데이터: {'✔ ' + str(csv) if csv.exists() else '✘ 없음 — data/ 밑에 있어야 한다'}")
if bad or not csv.exists():
    print("\n  아직 준비가 안 됐다:", ", ".join(bad) or "데이터 없음")
    sys.exit(1)
PY
RC=$?

if [ "$RC" = "0" ]; then
    printf '\n준비 끝. 실행: \033[1mbash run.sh\033[0m  (빠른 확인은 KAMP_FAST=1 bash run.sh)\n'
else
    cat <<'EOS'

── 막혔을 때 ─────────────────────────────────────────────
 Permission denied / OSError: [Errno 13]
   → 공유 환경에 쓰기 권한이 없다. 다음 중 하나로 다시 실행:
       bash setup_pod.sh            (--user 로 자동 전환됨)
       bash setup_pod.sh --venv     (홈에 가상환경을 만들어 거기에 설치)

 --venv 로 깔았다면 주피터에서 커널을 바꿔야 한다:
       python -m ipykernel install --user --name kamp-exp \
              --display-name "KAMP 자원최적화 (실험전용)"
   그 뒤 노트북 우상단에서 그 커널을 선택.

 폐쇄망이라 pip 가 아예 안 된다면
   → 관리자에게 requirements.txt 를 주고 설치를 요청하거나,
     이미 깔린 버전으로 그냥 돌려본다(대부분 동작한다).
──────────────────────────────────────────────────────────
EOS
fi
exit $RC
