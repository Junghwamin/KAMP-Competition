#!/usr/bin/env bash
# 포드 환경 진단 — 읽기만 한다. 아무것도 설치하지 않고 아무 파일도 고치지 않는다.
#
#   bash check_env.sh
#
# 출력 전체를 그대로 복사해서 보내주면 설치 전략을 정할 수 있다.
cd "$(dirname "$0")" 2>/dev/null || true

echo "════════ 1. 기본 ════════"
echo "whoami : $(id -un) (uid=$(id -u))"
echo "host   : $(hostname)"
echo "cwd    : $(pwd)"
echo "python : $(command -v python) / $(python -V 2>&1)"
echo "pip    : $(python -m pip --version 2>&1 | head -1)"
echo "conda  : ${CONDA_DEFAULT_ENV:-없음} / $(command -v conda || echo 'conda 명령 없음')"

echo
echo "════════ 2. 설치 경로와 권한 ════════"
python - <<'PY'
import site, sys, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(f"site-packages : {sp}")
print(f"  쓰기 가능   : {os.access(sp, os.W_OK)}")
print(f"user site     : {site.getusersitepackages()}")
print(f"  USER_SITE 활성 : {site.ENABLE_USER_SITE}")
print(f"sys.prefix    : {sys.prefix}")
print(f"base_prefix   : {sys.base_prefix}  (다르면 이미 venv 안)")
PY
echo "홈 디렉터리 쓰기: $([ -w "$HOME" ] && echo 가능 || echo 불가)"
echo "홈 용량:"; df -h "$HOME" 2>/dev/null | tail -1

echo
echo "════════ 3. 필요한 패키지 현황 ════════"
python - <<'PY'
import importlib.metadata as md
need = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "lightgbm", "optuna", "statsmodels", "tensorflow", "shap",
        "ipykernel", "nbconvert", "jupyterlab"]
imp = {"scikit-learn": "sklearn", "jupyterlab": "jupyterlab"}
for p in need:
    try:
        v = md.version(p)
        print(f"  ✔ {p:14s} {v}")
    except Exception:
        print(f"  ✘ {p:14s} 없음")
PY

echo
echo "════════ 4. 한글 폰트 ════════"
echo "fc-list 한국어 폰트:"
fc-list :lang=ko 2>/dev/null | head -5 || echo "  fc-list 명령 없음"
echo "matplotlib 이 인식하는 한글 후보:"
python - <<'PY'
try:
    from matplotlib import font_manager
    names = sorted({f.name for f in font_manager.fontManager.ttflist})
    hit = [n for n in names
           if any(k in n.lower() for k in ("nanum", "noto", "cjk", "gothic", "malgun", "gulim", "batang"))]
    print("  ", hit if hit else "없음 — 그림의 한글이 깨진다")
    print("   (노트북이 찾는 이름: Malgun Gothic / NanumGothic / Noto Sans KR / AppleGothic)")
    print(f"   총 폰트 {len(names)}종")
except Exception as e:
    print("  matplotlib 확인 실패:", e)
PY

echo
echo "════════ 5. GPU ════════"
command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || echo "  nvidia-smi 없음"
python - <<'PY'
try:
    import tensorflow as tf
    print("  tensorflow", tf.__version__, "· GPU:", tf.config.list_physical_devices("GPU") or "없음")
except Exception as e:
    print("  tensorflow 확인 불가:", type(e).__name__, e)
PY

echo
echo "════════ 6. 데이터 파일 ════════"
ls -la data/okm_augumented_2021.csv 2>/dev/null || echo "  data/okm_augumented_2021.csv 없음"
ls -la 자원최적화_실험전용.ipynb 2>/dev/null || echo "  노트북 없음"

echo
echo "════════ 7. 네트워크(pypi) ════════"
timeout 20 python -m pip download --no-deps --dest /tmp/_pipchk pip -q 2>&1 | tail -2 \
  && echo "  pypi 접근 가능" || echo "  pypi 접근 실패(폐쇄망일 수 있다)"
rm -rf /tmp/_pipchk 2>/dev/null || true

echo
echo "진단 끝 — 위 출력을 전부 복사해서 보내주세요."
