#!/usr/bin/env python3
"""`src/*.py` → **실험 전용 단일 노트북** 조립.

`tools/build_notebook.py` 가 만드는 제출용 노트북과 같은 소스에서 조립하되,
**보고서·제출물 생성 부분을 빼고**, **정의를 앞단에 모으고 실험만 뒤에 남긴다**.
`data/okm_augumented_2021.csv` 와 이 노트북 하나만 있으면 실행된다.

구조
----
    [1부] 환경과 정의   — 임포트 · 상수 · 함수 정의 전부
    [2부] 실험          — 데이터가 흐르는 실행문과 해설만

무엇을 앞단으로 올리는가
------------------------
임포트 · 함수/클래스 정의 · **그 시점에 이미 계산 가능한 상수 대입**만 올린다.
`THETA = compute_theta(df)` 처럼 **프로젝트 함수를 호출하거나 런타임 데이터에 기대는
대입은 올리지 않는다** — 올리면 `df` 보다 먼저 실행되어 죽는다. 이 판정은 손이 아니라
이름 해석으로 기계가 한다(`front_safe`).

안전장치 (하나라도 걸리면 빌드 실패)
-----------------------------------
1. 뺀 절이 정의하던 이름을 남은 코드가 쓰는가
2. stage 간 `from sNN_ import` 가 남았는가
3. **원본 최상위 문장이 하나도 빠지거나 중복되지 않았는가** (줄 단위 대조)
4. 앞단 문장이 뒷단에서만 정의되는 이름을 쓰는가 (실행 순서 역전)

    python tools/build_experiment_notebook.py
    python tools/build_experiment_notebook.py --layout chapter   # 원본 장 순서 그대로
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re
import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mindmap_canvas import parse_module  # noqa: E402  (같은 섹션 파서를 재사용)

STRIP_TAG = "nb-strip"
DEFAULT_OUT = "자원최적화_실험전용.ipynb"
RE_STAGE_IMPORT = re.compile(r"^\s*from\s+s\d\d_\w+\s+import", re.M)
ENV_MODULE = "s00_env"          # 이 파일은 통째로 '환경과 정의'

# 모듈 -> {섹션키: 빼는 이유}. "*" 는 파일 전체.
EXCLUDE: dict[str, dict[str, str]] = {
    "s06_eval": {
        "6.7": "보고서 서술 교정 — 10.2절 문장 치환사전 입력용",
    },
    "s08_simulation": {
        "8.5": "보고서 4장 본문 초안(md) 생성",
    },
    "s09_creativity": {
        "*": "창의성·차별성 근거 정리 + 보고서 5장 초안 — 실험이 아니라 보고서 재료",
    },
    "s10_package": {
        "10.2": "보고서 채움표 · 문장 치환사전 · 장별 PASS/FAIL",
        "10.3": "report_tbd_filled.md 생성",
        "10.4": "requirements.txt · README(6장 본문) 생성",
        "10.6": "발표자료 골격(pptx) · 개인정보 스캔",
        "최종 게이트 — 제출 준비 상태 점검": "제출 준비 점검",
    },
}

INTRO = """# 자원최적화 제안모델 — 실험 전용 단일 노트북

`data/okm_augumented_2021.csv` 와 **이 노트북 하나만** 있으면 처음부터 끝까지 실행된다.
`src/*.py` 에서 자동 조립된다 (생성기: `tools/build_experiment_notebook.py`).

| | 내용 |
|---|---|
| **1부 환경과 정의** | 임포트 · 경로 · 시드 · 한글폰트 · 저장 유틸 · **모든 함수 정의** |
| **2부 실험** | 데이터 적재 → 진단 → 피처 → 분할 → 베이스라인 → 모델 → 평가 → 해석 → 시뮬레이션 → 예측파일 |

- 작업 디렉터리를 이 노트북이 있는 폴더로 두고 실행한다(경로가 상대경로다).
- 산출물은 `outputs/figures`, `outputs/tables` 에 저장된다.
- 빠르게 한 번 돌려보려면 실행 전 환경변수 `KAMP_FAST=1` (탐색·에폭만 축소, 코드는 동일).

**뺀 절** — 보고서·제출물 생성 부분이며 실험 결과에는 영향이 없다:
{dropped}
"""

PART1 = """---
# 1부 · 환경과 정의

여기서는 **아무 실험도 하지 않는다.** 임포트, 경로·시드·폰트 설정, 저장 유틸,
그리고 2부에서 쓸 함수를 전부 정의만 한다. 위에서 아래로 한 번 실행하고 2부로 간다.
"""

PART2 = """---
# 2부 · 실험

여기서부터 데이터가 흐른다. 각 셀은 1부에서 정의한 함수를 부르고 결과를 남긴다.
"""


# ──────────────────────────────────────────────────────────────────────
# 소스 분해
# ──────────────────────────────────────────────────────────────────────
def cell_spans(lines: list[str]) -> list[tuple[str, int, int]]:
    """percent 셀 경계를 (종류, 시작행, 끝행) 으로 끊는다. 시작행은 `# %%` 마커."""
    marks = [i for i, l in enumerate(lines, 1) if l.startswith("# %%")]
    out = []
    for j, m in enumerate(marks):
        end = marks[j + 1] - 1 if j + 1 < len(marks) else len(lines)
        kind = "markdown" if "[markdown]" in lines[m - 1] else "code"
        out.append((kind, m, end))
    return out


def cell_start_index(marks: list[int], line: int) -> int:
    hit = [s for s in marks if s <= line]
    return hit[-1] if hit else 1


def dropped_lines(path: Path, lines: list[str]) -> tuple[set[int], list[str]]:
    """제외 섹션이 차지하는 행 번호 집합과 설명."""
    spec = EXCLUDE.get(path.stem, {})
    if not spec:
        return set(), []
    if "*" in spec:
        return set(range(1, len(lines) + 1)), [f"- `{path.stem}` 전체 — {spec['*']}"]

    mod = parse_module(path)
    keys = [s.key for s in mod.sections]
    unknown = [k for k in spec if k not in keys]
    if unknown:
        raise SystemExit(f"{path.name}: 존재하지 않는 섹션키 {unknown}\n  실제 섹션: {keys}")

    marks = [i for i, l in enumerate(lines, 1) if l.startswith("# %%")]
    out, notes = set(), []
    for i, sec in enumerate(mod.sections):
        if sec.key not in spec:
            continue
        a = cell_start_index(marks, sec.start)
        b = (cell_start_index(marks, mod.sections[i + 1].start) - 1
             if i + 1 < len(mod.sections) else len(lines))
        out.update(range(a, b + 1))
        label = f"{sec.num} {sec.title}".strip() if sec.num else sec.title
        notes.append(f"- `{path.stem}` **{label}** — {spec[sec.key]}")
    return out, notes


def block_of(stmt: ast.stmt, lines: list[str]) -> tuple[int, int]:
    """문장이 차지하는 행 범위. 바로 위에 붙어 있는 주석줄까지 포함한다."""
    start = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])])
    i = start - 1
    while i >= 1:
        prev = lines[i - 1]
        if prev.startswith("# %%") or not prev.lstrip().startswith("#"):
            break
        start = i
        i -= 1
    return start, stmt.end_lineno or stmt.lineno


def defined_names(stmt: ast.stmt) -> set[str]:
    out = set()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            out.add((a.asname or a.name).split(".")[0])
    elif isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            out.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        out.add(stmt.target.id)
    return out


def mutated_names(trees: list[ast.AST]) -> set[str]:
    """나중에 내용이 바뀔 수 있는 이름 — 앞단에서 읽으면 '빈 값'을 집게 된다.

    `acc = []` 을 앞단에 올린 뒤 `tbl = pd.DataFrame(acc)` 까지 올려버리면
    루프가 채우기 전의 빈 리스트를 집는다. 예외도 안 나고 결과만 틀린다.
    """
    out: set[str] = set()
    for tree in trees:
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    root = t
                    while isinstance(root, (ast.Subscript, ast.Attribute)):
                        root = root.value
                    if isinstance(root, ast.Name) and root is not t:
                        out.add(root.id)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)):
                out.add(n.func.value.id)      # 메서드 호출 수신자
    return out


def front_safe(value: ast.AST, front: set[str], project_funcs: set[str],
               mutated: set[str], imports: set[str]) -> bool:
    """이 식을 '지금 시점'에 평가해도 되는가.

    - 쓰이는 이름이 전부 앞단에 이미 있어야 한다
    - **프로젝트 함수 호출이 들어 있으면 안 된다** (런타임 전역에 기댈 수 있다)
    - **나중에 내용이 바뀌는 이름에 기대면 안 된다** (모듈은 예외)
    """
    for n in ast.walk(value):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in front and n.id not in dir(builtins):
                return False
            if n.id in mutated and n.id not in imports:
                return False
        if isinstance(n, ast.Call):
            root = n.func
            while isinstance(root, ast.Attribute):
                root = root.value
            rid = getattr(root, "id", None)
            if rid is None:
                return False
            if isinstance(n.func, ast.Name) and rid in project_funcs:
                return False
    return True


def plan_file(path: Path, front: set[str], project_funcs: set[str],
              mutated: set[str], imports: set[str]):
    """파일 하나를 (앞단 블록, 제외 행, 설명) 으로 나눈다. `front` 는 갱신된다."""
    lines = path.read_text(encoding="utf-8").splitlines()
    drop, notes = dropped_lines(path, lines)
    tree = ast.parse("\n".join(lines))
    front_blocks: list[tuple[int, int]] = []

    for stmt in tree.body:
        if stmt.lineno in drop:
            continue
        movable = False
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            movable = not RE_STAGE_IMPORT.match(lines[stmt.lineno - 1])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            movable = True
        elif isinstance(stmt, ast.Assign) and stmt.value is not None:
            movable = (all(isinstance(t, ast.Name) for t in stmt.targets)  # 변형 대입 금지
                       and front_safe(stmt.value, front, project_funcs, mutated, imports))
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            movable = (isinstance(stmt.target, ast.Name)
                       and front_safe(stmt.value, front, project_funcs, mutated, imports))
        if movable:
            front_blocks.append(block_of(stmt, lines))
            front |= defined_names(stmt)
    return lines, front_blocks, drop, notes


def group_blocks(blocks: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """행이 이어지는 블록끼리 한 셀로 묶는다."""
    groups: list[list[tuple[int, int]]] = []
    for b in sorted(blocks):
        if groups and b[0] <= groups[-1][-1][1] + 2:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


# ──────────────────────────────────────────────────────────────────────
# 셀 만들기
# ──────────────────────────────────────────────────────────────────────
def code_cell(src: str):
    c = nbformat.v4.new_code_cell(src.rstrip())
    c["metadata"] = {}
    return c


def cells_from_percent(src: str) -> list:
    nb = jupytext.reads(src, fmt="py:percent")
    out = []
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", []) or []
        if STRIP_TAG in tags or not cell.get("source", "").strip():
            continue
        cell["metadata"].pop("lines_to_next_cell", None)
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        out.append(cell)
    return out


def names_in(src: str) -> tuple[set[str], set[str]]:
    """(정의되는 이름, 쓰이는 이름) — 과대추정. 누락 탐지용."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()
    d, u = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (d if isinstance(n.ctx, (ast.Store, ast.Del)) else u).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                d.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg):
            d.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            d.add(n.name)
    return d, u


def toplevel_defined(src: str) -> set[str]:
    """**최상위에서** 정의되는 이름만. 함수 매개변수·지역변수를 섞으면
    `df` 같은 흔한 이름이 '이미 있다'로 잡혀 순서 검증이 통째로 무력해진다."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for n in tree.body:
        out |= defined_names(n)
    return out


def chapter_label(path: Path) -> str:
    """`## 2. 파생변수 …` 를 찾아 짧은 이름으로."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ## "):
            return line[5:].strip()
    return path.stem


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 전용 단일 노트북 조립")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--layout", choices=["defs-first", "chapter"], default="defs-first",
                    help="defs-first=정의를 1부로 모음 · chapter=원본 장 순서 그대로")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(SRC_DIR.glob("s[0-9][0-9]_*.py"))
    if not paths:
        print("src 에 stage 파일이 없습니다.", file=sys.stderr)
        return 2

    trees = [ast.parse(p.read_text(encoding="utf-8")) for p in paths]
    project_funcs: set[str] = set()
    imports: set[str] = set()
    for t in trees:
        for n in t.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                project_funcs.add(n.name)
        for n in ast.walk(t):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                imports.update((a.asname or a.name).split(".")[0] for a in n.names)
    mutated = mutated_names(trees)

    front_names: set[str] = set()
    front_cells, exp_cells, notes = [], [], []
    kept_src, cut_src = [], []
    stat = []

    for p in paths:
        defs_first = args.layout == "defs-first"
        if defs_first and p.stem == ENV_MODULE:
            # 환경 파일은 통째로 1부 (마크다운 해설도 그대로 가져간다)
            lines = p.read_text(encoding="utf-8").splitlines()
            drop, note = dropped_lines(p, lines)
            notes += note
            keep = [l for i, l in enumerate(lines, 1) if i not in drop]
            front_cells += cells_from_percent("\n".join(keep) + "\n")
            front_names |= toplevel_defined("\n".join(keep))
            kept_src.append("\n".join(keep))
            cut_src.append("\n".join(l for i, l in enumerate(lines, 1) if i in drop))
            stat.append((p.stem, len(lines), "전체 1부"))
            continue

        lines, blocks, drop, note = (
            plan_file(p, front_names, project_funcs, mutated, imports) if defs_first
            else (p.read_text(encoding="utf-8").splitlines(), [],
                  *dropped_lines(p, p.read_text(encoding="utf-8").splitlines())))
        notes += note
        moved = {i for a, b in blocks for i in range(a, b + 1)}

        if blocks:
            front_cells.append(nbformat.v4.new_markdown_cell(
                f"### 정의 — {chapter_label(p)}"))
            for g in group_blocks(blocks):
                src = "\n".join("\n".join(lines[a - 1:b]) for a, b in g)
                front_cells.append(code_cell(src))

        # 2부: 앞단으로 올라간 줄과 제외된 줄을 뺀 나머지
        rest = []
        for kind, a, b in cell_spans(lines):
            body = [l for i, l in enumerate(lines[a:b], a + 1)
                    if i not in moved and i not in drop]
            if a in drop:
                continue
            if any(x.strip() and not x.strip().startswith("#") for x in body) or kind == "markdown":
                if kind == "markdown" and not any(x.strip() for x in body):
                    continue
                rest.append(lines[a - 1])
                rest.extend(body)
        exp_cells += cells_from_percent("\n".join(rest) + "\n")

        keep_all = "\n".join(l for i, l in enumerate(lines, 1) if i not in drop)
        kept_src.append(keep_all)
        cut_src.append("\n".join(l for i, l in enumerate(lines, 1) if i in drop))
        stat.append((p.stem, len(lines), f"1부 {len(blocks)}문장"))

    kept_all, cut_all = "\n".join(kept_src), "\n".join(cut_src)

    # ── 안전장치 1: 뺀 절이 정의하던 이름을 남은 코드가 쓰는가 ──────────
    _, kuse = names_in(kept_all)
    kdef = toplevel_defined(kept_all)
    cdef = toplevel_defined(cut_all)
    broken = sorted((kuse & cdef) - kdef - set(dir(builtins)))
    if broken:
        print("조립 실패 — 뺀 절이 정의하던 이름을 남은 코드가 씁니다:", file=sys.stderr)
        for b in broken:
            print(f"  ✗ {b}", file=sys.stderr)
        return 1

    cells = front_cells + exp_cells if args.layout == "defs-first" else exp_cells
    if args.layout == "defs-first":
        cells = ([nbformat.v4.new_markdown_cell(PART1)] + front_cells
                 + [nbformat.v4.new_markdown_cell(PART2)] + exp_cells)

    # ── 안전장치 2: stage import 잔존 ─────────────────────────────────
    leftover = [c for c in cells
                if c.cell_type == "code" and RE_STAGE_IMPORT.search(c["source"])]
    if leftover:
        print(f"조립 실패 — stage import 가 {len(leftover)}개 남았습니다:", file=sys.stderr)
        for c in leftover[:5]:
            print("  ✗ " + c["source"].splitlines()[0][:70], file=sys.stderr)
        return 1

    # ── 안전장치 3: 원본 문장 보존 (한 줄도 잃거나 중복되지 않는다) ────
    asm = "\n".join(c["source"] for c in cells if c.cell_type == "code")
    asm_stmts = [ast.dump(n) for n in ast.parse(asm).body]
    src_stmts = []
    for p in paths:
        lines = p.read_text(encoding="utf-8").splitlines()
        drop, _ = dropped_lines(p, lines)
        for n in ast.parse("\n".join(lines)).body:
            if n.lineno in drop:
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)) and RE_STAGE_IMPORT.match(
                    lines[n.lineno - 1]):
                continue
            src_stmts.append(ast.dump(n))
    if sorted(asm_stmts) != sorted(src_stmts):
        lost = len(src_stmts) - len(asm_stmts)
        print(f"조립 실패 — 최상위 문장 수가 다릅니다 (원본 {len(src_stmts)} vs "
              f"조립 {len(asm_stmts)}, 차 {lost})", file=sys.stderr)
        from collections import Counter
        d = Counter(src_stmts) - Counter(asm_stmts)
        for k in list(d)[:5]:
            print("  ✗ 누락: " + k[:100], file=sys.stderr)
        d2 = Counter(asm_stmts) - Counter(src_stmts)
        for k in list(d2)[:5]:
            print("  ✗ 중복/추가: " + k[:100], file=sys.stderr)
        return 1

    # ── 안전장치 4: 1부가 2부에서만 정의되는 이름을 쓰는가 ─────────────
    if args.layout == "defs-first":
        fsrc = "\n".join(c["source"] for c in front_cells if c.cell_type == "code")
        esrc = "\n".join(c["source"] for c in exp_cells if c.cell_type == "code")
        fdef = toplevel_defined(fsrc)
        edef = toplevel_defined(esrc)
        # 함수 '몸통' 안의 이름은 호출 시점에 풀리므로 제외한다
        top = ast.parse(fsrc)
        runtime_free = set()
        for n in top.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for x in ast.walk(n):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    runtime_free.add(x.id)
        bad = sorted((runtime_free & edef) - fdef - set(dir(builtins)))
        if bad:
            print("조립 실패 — 1부가 2부에서 정의되는 이름을 씁니다(실행 순서 역전):",
                  file=sys.stderr)
            for b in bad:
                print(f"  ✗ {b}", file=sys.stderr)
            return 1

    intro = nbformat.v4.new_markdown_cell(
        INTRO.format(dropped="\n".join(notes) if notes else "- (없음)"))
    nb = nbformat.v4.new_notebook(cells=[intro] + cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.9"}

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"셀 {len(nb.cells)} (코드 {n_code} · 마크다운 {len(nb.cells) - n_code})"
          + (f" · 1부 {len(front_cells)} / 2부 {len(exp_cells)}"
             if args.layout == "defs-first" else ""))
    for stem, nl, how in stat:
        print(f"  {stem:15s} {nl:5d}줄  {how}")
    print(f"뺀 절 {len(notes)}개:")
    for n in notes:
        print("  " + n.replace("- ", "", 1))
    print("검증 통과: 이름 의존성 · stage import · 문장 보존 · 실행순서 4종")

    if args.dry_run:
        print("(dry-run — 파일을 쓰지 않았습니다)")
        return 0

    out = ROOT / args.out
    nbformat.write(nb, out)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
    raise SystemExit(main())
