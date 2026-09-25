"""포드 제출용 단일 노트북 패키지를 만든다 (2026-09-25 에 만든 pod_실험전용_v2 와 같은 구성).

구성: 노트북 1개(생성기로 src/ 에서 조립) + data/ + 실행 스크립트(pod_files/) + serving/(선택 실행).
폴더와 같은 이름의 .zip 도 만든다. 점(.)으로 시작하는 폴더·__pycache__ 는 넣지 않는다
(도구 상태 폴더 .omc/ 에 계정명이 들어간 적이 있다).

    python -X utf8 experiments/2026-09-25_pod_package/make_pod_package.py --out <만들 폴더>
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = "자원최적화_실험전용.ipynb"
DATA = Path("data") / "okm_augumented_2021.csv"
POD_FILES = ["README.md", "run.sh", "check_env.sh", "setup_pod.sh", "requirements.txt"]
SERVING_FILES = ["__init__.py", "__main__.py", "_core.py", "api.py", "bundle.py", "cli.py", "contract.py",
                 "errors.py", "pipeline.py", "README.md", "requirements.txt", "config/holidays_kr.json"]


def privacy_hits(root: Path) -> list[str]:
    """csv 를 뺀 텍스트 파일에서 계정명·사용자 절대경로를 찾는다."""
    pats = [re.escape(Path.home().name), r"[A-Za-z]:\\+[Uu]sers", r"[A-Za-z]:/+[Uu]sers", r"/home/[A-Za-z]"]
    hits = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() in (".csv", ".png", ".zip"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        hits += [f"{f.relative_to(root)}: {p}" for p in pats if re.search(p, text)]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="포드 제출 패키지 조립")
    ap.add_argument("--out", required=True, help="만들 패키지 폴더 (없거나 비어 있어야 한다)")
    out = Path(ap.parse_args().out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"이미 내용이 있는 폴더다: {out.name}")
    (out / "data").mkdir(parents=True, exist_ok=True)

    # 1) 노트북 — 제출 노트북과 같은 src/ 에서 조립 (생성기의 안전장치 4종이 통과해야 한다)
    subprocess.run([sys.executable, "-X", "utf8", str(REPO / "tools" / "build_experiment_notebook.py"),
                    "--out", str(out / NOTEBOOK)], check=True)
    # 2) 데이터 · 실행 스크립트 · serving
    shutil.copy2(REPO / DATA, out / DATA)
    for f in POD_FILES:
        shutil.copy2(HERE / "pod_files" / f, out / f)
    for f in SERVING_FILES:
        (out / "serving" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "serving" / f, out / "serving" / f)

    hits = privacy_hits(out)
    if hits:
        print("개인정보 의심:", *hits, sep="\n  ")
        return 1

    # 3) zip — 점 폴더·__pycache__ 제외
    zpath = out.with_suffix(".zip")
    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(out).parts))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, (Path(out.name) / f.relative_to(out)).as_posix())
    print(f"패키지 {len(files)}개 파일 → {out.name}/ · {zpath.name} ({zpath.stat().st_size / 1e6:.2f} MB) · 개인정보 0건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
