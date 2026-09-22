"""src/*.py (jupytext percent 포맷) → 자원최적화_제안모델.ipynb 조립.

설계 원칙
---------
코드는 **한 벌만** 존재한다. `src/sNN_*.py` 가 단일 진실원이고
`.ipynb` 는 거기서 조립된 빌드 산출물이다. 노트북에 붙여넣은 사본을
따로 유지하지 않는다(반드시 드리프트한다).

변환 규칙
---------
1. `# %% tags=["nb-strip"]` 로 표시된 셀은 노트북에서 **삭제**한다.
   이 셀들은 stage 모듈 간 변수 전달용 import 문으로, 노트북에서는
   동일 커널에 변수가 이미 존재하므로 불필요하다.
2. 그 외 코드/마크다운 셀은 파일 순서 그대로 이어붙인다.
3. 셀 출력은 비운다(실행은 nbconvert --execute 가 담당).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import jupytext
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES_DIR = PROJECT_ROOT / "src"
OUTPUT_NB = PROJECT_ROOT / "자원최적화_제안모델.ipynb"

STRIP_TAG = "nb-strip"

# 조립 순서 (파일명 접두 sNN 순)
STAGE_GLOB = "s[0-9][0-9]_*.py"


def load_stage_cells(path: Path) -> list:
    """percent 포맷 .py 를 읽어 nbformat 셀 리스트로 반환한다.

    Parameters
    ----------
    path : Path
        jupytext percent 포맷 파이썬 파일.

    Returns
    -------
    list
        nb-strip 태그 셀이 제거된 nbformat 셀 리스트.
    """
    nb = jupytext.read(path, fmt="py:percent")
    kept = []
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", []) or []
        if STRIP_TAG in tags:
            continue  # stage 간 import 문 — 노트북에서는 불필요
        # 빈 셀 제거
        if not cell.get("source", "").strip():
            continue
        cell["metadata"].pop("lines_to_next_cell", None)
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        kept.append(cell)
    return kept


def build(stage_files: list[Path] | None = None) -> nbformat.NotebookNode:
    """stage 파일들을 하나의 노트북으로 조립한다."""
    if stage_files is None:
        stage_files = sorted(STAGES_DIR.glob(STAGE_GLOB))
    if not stage_files:
        raise FileNotFoundError(f"stage 파일이 없습니다: {STAGES_DIR}/{STAGE_GLOB}")

    cells = []
    for f in stage_files:
        cells.extend(load_stage_cells(f))

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.9"}
    return nb


def validate(nb: nbformat.NotebookNode) -> dict:
    """조립된 노트북의 구조를 점검한다.

    Returns
    -------
    dict
        n_cells / n_code / n_md / chapters / subsections / strays
    """
    md_sources = [c.source for c in nb.cells if c.cell_type == "markdown"]
    code_sources = [c.source for c in nb.cells if c.cell_type == "code"]

    chapters = [s for s in md_sources if re.match(r"^##\s+\d+\.", s.strip())]
    subsections = [s for s in md_sources if re.match(r"^###\s+\d+\.\d+", s.strip())]

    # 노트북에 남으면 안 되는 흔적.
    # 주석·문자열에서의 '언급'은 위반이 아니다. 예를 들어 4.4절 결함표는
    # `freq='H'` 를 설명 문구로 담고 있고, 0.3절 주석은 `os.getcwd()` 를
    # "쓰지 않는다" 는 설명으로 언급한다. 따라서 주석 줄을 걷어내고
    # **실제 호출 형태**만 검사한다.
    strays = []
    for src in code_sources:
        code = "\n".join(
            ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
        )
        for pat, why in [
            (r"^from s\d\d_\w+ import", "stage 간 import 잔존"),
            (r"^(?!\s*[\"']).*\bos\.getcwd\(\)", "CWD 출력 금지(블라인드)"),
            (r"C:\\\\Users", "절대경로 잔존(블라인드)"),
            (
                r"(?:date_range|resample|period_range|Grouper)\s*\([^)]*freq\s*=\s*['\"]H['\"]",
                "pandas 3.x 크래시 (freq='H')",
            ),
            (r"\.twinx\(\)", "이중 축 금지"),
        ]:
            if re.search(pat, code, re.MULTILINE):
                strays.append(why)

    return {
        "n_cells": len(nb.cells),
        "n_code": len(code_sources),
        "n_md": len(md_sources),
        "chapters": len(chapters),
        "subsections": len(subsections),
        "strays": sorted(set(strays)),
    }


def main() -> int:
    nb = build()
    info = validate(nb)
    if info["strays"]:
        print("[FAIL] 노트북에 남으면 안 되는 흔적:", file=sys.stderr)
        for s in info["strays"]:
            print(f"   - {s}", file=sys.stderr)
        return 1
    nbformat.write(nb, OUTPUT_NB)
    print(f"[OK] {OUTPUT_NB.name}")
    print(
        f"     셀 {info['n_cells']} (코드 {info['n_code']} / 마크다운 {info['n_md']}) | "
        f"대주제 {info['chapters']} | 소주제 {info['subsections']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
