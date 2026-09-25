# %% [markdown]
# # START HERE — 실행 전 환경 점검
#
# 이 노트북을 **제일 먼저** 전체 실행(`Run All`)하세요. 지금 이 환경에서
# 메인 노트북이 돌아갈지 판정하고, 문제가 있으면 무엇을 해야 하는지 알려줍니다.
#
# | | |
# |---|---|
# | 걸리는 시간 | 30초 |
# | 메인 노트북 | `자원최적화_제안모델.ipynb` (FULL 실행 시 CPU 30~60분) |
# | GPU | **필요 없습니다** (아래 5번 참조) |
#
# 점검에 실패하면 `04_TROUBLESHOOTING.md` 를 보세요.

# %% [markdown]
# ## 1. 작업 디렉터리 확인
#
# 메인 노트북은 `data/...`, `outputs/...` 같은 **상대경로**를 씁니다.
# 따라서 작업 디렉터리가 이 패키지 루트여야 합니다.

# %%
from pathlib import Path
import sys

FAIL: list[str] = []

DATA = Path("data") / "okm_augumented_2021.csv"
if DATA.exists():
    print(f"OK   데이터 파일 발견: {DATA}  ({DATA.stat().st_size:,} bytes)")
else:
    FAIL.append(
        "작업 디렉터리가 패키지 루트가 아니다."
        "\n      Jupyter 에서 이 노트북이 있는 폴더를 열었는지 확인하라."
        "\n      터미널이면: cd <이 폴더> 후 다시 실행"
    )
    print("FAIL 데이터 파일을 찾을 수 없다")

for _d in ("src", "tools", "tests", "outputs"):
    _exists = Path(_d).is_dir()
    print(f"{'OK  ' if _exists else '--  '} {_d}/ "
          f"{'존재' if _exists else '없음 (선택 자산이므로 실행에는 무관)'}")

# %% [markdown]
# ## 2. 패키지 버전 대조
#
# 재현성 검증(1,988개 셀 소수 6자리 일치)이 **아래 조합**에서 이뤄졌습니다.
# 버전이 다르면 수치가 달라질 수 있어 불일치를 실패로 처리합니다.
#
# `pandas 3.0.5` 가 특히 중요합니다 — 무음실패 탐지(`ChainedAssignmentError` 승격)가
# 3.x 동작에 의존합니다. pod 이미지에 2.x 가 남아 있으면 여기서 걸립니다.

# %%
import importlib

EXPECT = {
    "numpy": "2.3.5", "pandas": "3.0.5", "sklearn": "1.9.1", "scipy": "1.17.1",
    "matplotlib": "3.11.1", "lightgbm": "4.7.0", "optuna": "5.0.0",
    "shap": "0.51.0", "statsmodels": "0.14.6",
}
_mismatch = []
for _mod, _want in EXPECT.items():
    try:
        _got = importlib.import_module(_mod).__version__
    except Exception as _e:
        _mismatch.append(f"{_mod}: import 실패 ({type(_e).__name__})")
        print(f"FAIL {_mod:<13s} import 실패")
        continue
    if _got == _want:
        print(f"OK   {_mod:<13s} {_got}")
    else:
        _mismatch.append(f"{_mod}: {_got} (기대 {_want})")
        print(f"FAIL {_mod:<13s} {_got}   <- 기대 {_want}")

_pv = ".".join(map(str, sys.version_info[:3]))
_pyok = sys.version_info[:2] in ((3, 10), (3, 11), (3, 12))
print(f"{'OK  ' if _pyok else 'WARN'} {'python':<13s} {_pv}  (검증 3.11.9 / 허용 3.10~3.12)")

if _mismatch:
    FAIL.append(
        "패키지 버전 불일치: " + "; ".join(_mismatch)
        + "\n      해결: bash setup_pod.sh"
    )

# %% [markdown]
# ## 3. 한글 폰트 — 실제로 렌더되는지 확인
#
# 그림의 축·제목이 한국어입니다. 폰트가 없으면 □□□ 로 깨집니다.
#
# 코드의 폰트 폴백 체인은 `Malgun Gothic → NanumGothic → Noto Sans KR → AppleGothic` 입니다.
# `Malgun Gothic` 은 Windows 전용이므로 **Linux pod 에서는 NanumGothic** 이 잡혀야 합니다.
#
# 아래는 설치 목록만 보지 않고 **실제로 한글을 그려** glyph 누락 경고가 나는지 확인합니다.
# 목록에 있어도 렌더가 안 되는 경우가 있어서, 렌더 테스트가 더 믿을 수 있습니다.

# %%
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

matplotlib.use("Agg")

_installed = {f.name for f in font_manager.fontManager.ttflist}
FONT_CHAIN = ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic")
_found = [c for c in FONT_CHAIN if c in _installed]
print("폴백 체인 중 설치된 폰트:", _found or "없음")

_picked = _found[0] if _found else None
if _picked:
    plt.rcParams["font.family"] = _picked
plt.rcParams["axes.unicode_minus"] = False

with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    _fig, _ax = plt.subplots(figsize=(3.2, 1.0))
    _ax.set_title("한글 렌더 테스트 — 평균전력 최대수요")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _fig.canvas.draw()
    plt.close(_fig)
    _glyph = [str(w.message) for w in _caught if "missing from font" in str(w.message)]

if _picked and not _glyph:
    print(f"OK   한글 렌더 정상 (폰트: {_picked})")
else:
    FAIL.append(
        "한글 폰트가 없어 그림이 깨진다."
        "\n      해결(Linux): sudo apt-get install -y fonts-nanum"
        " && rm -rf ~/.cache/matplotlib"
        "\n      설치 후 **커널을 재시작**해야 적용된다"
    )
    print("FAIL 한글이 깨진다", f"(glyph 누락 {len(_glyph)}건)" if _glyph else "")

# %% [markdown]
# ## 4. 디스크 여유
#
# `outputs/` 를 다시 만드는 데 약 60MB 가 필요합니다 (그림 39장 + 표 114개 + 모델 번들 약 51MB).

# %%
import shutil

_total, _used, _free = shutil.disk_usage(".")
_gb = _free / 1024 ** 3
print(f"{'OK  ' if _gb > 1 else 'FAIL'} 여유 공간 {_gb:.1f} GB (필요 ~60MB, 권장 1GB 이상)")
if _gb <= 1:
    FAIL.append(f"디스크 여유가 부족하다 ({_gb:.1f} GB)")

# %% [markdown]
# ## 5. GPU — 인식 여부와 **쓰이지 않는다는 사실**
#
# > ### 이 파이프라인은 GPU 를 쓰지 않습니다.
# >
# > - 주력 모델 **LightGBM** 은 CPU 전용 빌드입니다.
# > - TensorFlow 모델은 **SimpleRNN(50유닛)** 과 **MLP(128-64)** 로, 학습 표본이
# >   약 5,600행뿐이어서 GPU 로 보내는 오버헤드가 이득보다 큽니다.
# >
# > GPU pod 에서도 문제없이 돌아가지만 **CPU pod 이 더 저렴합니다.**
# > 아래 출력은 참고용이며, GPU 가 0개여도 정상입니다.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf

    _gpus = tf.config.list_physical_devices("GPU")
    print(f"OK   tensorflow {tf.__version__} / GPU 인식 {len(_gpus)}개")
    if _gpus:
        print("     GPU 가 보이지만 이 파이프라인은 사용하지 않는다 (정상)")
    else:
        print("     CPU 모드로 실행된다 (정상, 권장)")
except Exception as _e:
    print(f"--   tensorflow 미설치/로드실패 ({type(_e).__name__})")
    print("     SimpleRNN·DNN 2개 모델만 건너뛰고 나머지는 정상 실행된다")
    print("     (TF 미설치 방어가 코드에 들어 있다)")

# %% [markdown]
# ## 6. 판정

# %%
RUN_GUIDE = """
다음 중 하나로 메인 노트북을 실행하라.

  A) Jupyter UI
     자원최적화_제안모델.ipynb 를 열고 Kernel -> Restart & Run All
     ※ FULL 실행은 CPU 에서 30~60분. 세션이 끊기면 처음부터 다시 돌려야 한다.

  B) 배치 실행 (권장 — 세션이 끊겨도 계속된다)
     bash run.sh run        (make 가 있으면 make run)
     로그 확인:  tail -f outputs/run.log

  C) 빠른 확인만 (수치는 최종값이 아님, 3~5분)
     bash run.sh run-fast   (make 가 있으면 make run-fast)

실행이 끝나면 반드시 (B·C 는 자동으로 수행한다):
     bash run.sh finalize   (make 가 있으면 make finalize)   # stderr 정리 + 개인정보 후스캔

결과 위치:
     outputs/figures/                   그림 39장
     outputs/tables/                    표 114개
     outputs/predictions_test_336h.csv  제출용 예측결과
     outputs/report_tbd_filled.md       보고서 채움표
     outputs/models/full/{eval,deploy}  모델 번들 (10.5절, git 제외)
"""

print("=" * 66)
if FAIL:
    print(f"판정: 실행 불가 — 해결할 문제 {len(FAIL)}건")
    print("=" * 66)
    for _i, _msg in enumerate(FAIL, 1):
        print(f"\n[{_i}] {_msg}")
    print("\n자세한 대처는 04_TROUBLESHOOTING.md 를 보라.")
    raise SystemExit("환경 점검 실패 — 위 문제를 해결한 뒤 이 노트북을 다시 실행하라")
else:
    print("판정: 실행 가능")
    print("=" * 66)
    print(RUN_GUIDE)
