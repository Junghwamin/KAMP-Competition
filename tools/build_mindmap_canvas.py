#!/usr/bin/env python3
"""`src/*.py` 를 Obsidian Canvas 마인드맵 + 미러 마크다운 노트로 변환한다.

계층은 LLM 요약이 아니라 **원문 정적 추출**로 만든다.

    모듈(`## N.`)  →  섹션(`### N.M`)  →  최상위 `def`

사용법
------
    python tools/build_mindmap_canvas.py --modules s02 --out "maps/_preview_s02.canvas"
    python tools/build_mindmap_canvas.py                      # 전체 → Call Graph.canvas

설계 문서: `_dev_docs/MINDMAP_PLAN.md`
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

# ── 레이아웃 상수 (px) ────────────────────────────────────────────────
W_ROOT, W_SEC, W_FUNC, W_IO = 440, 400, 460, 420
X_ROOT, X_SEC, X_FUNC = 0, 560, 1020
X_ART = X_FUNC + W_FUNC + 100          # 산출물 열 (오른쪽)
X_IN = -(W_IO + 140)                   # 입력 열 (왼쪽)
GAP_FUNC, GAP_SEC, GAP_MODULE, GAP_MODULE_X = 18, 56, 220, 260
GAP_ART = 14
PAD_GROUP, LABEL_SPACE = 48, 56
LINE_H, MIN_H, PAD_H = 23, 60, 26

COLOR_IN, COLOR_ART, COLOR_FLOW = "4", "3", "2"   # 입력=초록, 산출물=노랑, 흐름=주황

MODULE_COLORS = ["5", "1", "2", "3", "4", "6"]  # canvas 기본 색 1~6 순환

# ── 정규식 ────────────────────────────────────────────────────────────
RE_CELL = re.compile(r"^# %%")
RE_HEADER = re.compile(r"^(#{2,4})\s+(.*\S)\s*$")
RE_SEC_NUM = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")
RE_BULLET = re.compile(r"^-\s*\*\*(목적|산출물|보고서 대응절)\*\*\s*[:：]\s*(.*)$")
RE_TOPDEF = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RE_FID = re.compile(r"F\d{1,3}")          # 그림 id (F01 … F39)
ILLEGAL_FS = re.compile(r'[\\/:*?"<>|#\[\]^]')


# ── 데이터 모델 ───────────────────────────────────────────────────────
@dataclass
class IOItem:
    kind: str            # "read" | "write"
    path: str            # outputs/tables/ch2_ablation.csv
    label: str           # 카드에 쓸 이름
    detail: str          # 부가 설명 (그림 제목 등)
    module: str
    lineno: int
    owner: str = ""      # 이 I/O 를 매단 함수명 ("" 이면 섹션 직속)


@dataclass
class Func:
    name: str
    signature: str
    doc: str
    lineno: int
    end_lineno: int
    code: str
    calls: list[str] = field(default_factory=list)
    io: list[IOItem] = field(default_factory=list)


@dataclass
class Section:
    num: str
    title: str
    heading: str
    start: int
    end: int
    md: list[str] = field(default_factory=list)
    purpose: str = ""
    outputs: str = ""
    artifacts: list[str] = field(default_factory=list)
    funcs: list[Func] = field(default_factory=list)
    io: list[IOItem] = field(default_factory=list)   # 섹션 직속 (함수 밖) I/O
    note_path: str = ""

    @property
    def key(self) -> str:
        return self.num or self.title

    @property
    def all_io(self) -> list[IOItem]:
        return self.io + [i for f in self.funcs for i in f.io]


@dataclass
class Module:
    stem: str
    path: Path
    title: str
    nlines: int
    sections: list[Section] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    dep_names: dict = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def io(self) -> list[IOItem]:
        return [i for s in self.sections for i in s.all_io]

    @property
    def funcs(self) -> list[Func]:
        return [f for s in self.sections for f in s.funcs]


# ── 유틸 ──────────────────────────────────────────────────────────────
def nid(*parts: str) -> str:
    """노드 id — 내용이 아니라 '정체성'으로 만든다(재생성해도 불변)."""
    return hashlib.sha1("\u0000".join(parts).encode("utf-8")).hexdigest()[:16]


def dispw(s: str) -> int:
    """동아시아 전각 문자를 2칸으로 세는 표시 너비."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


RE_WIKI_ALIAS = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")


def rendered(line: str) -> str:
    """높이는 '보이는 글자' 기준으로 잰다 — 위키링크 경로는 렌더되지 않는다."""
    line = RE_WIKI_ALIAS.sub(r"\1", line)
    return line.replace("**", "").replace("`", "").lstrip("#").lstrip()


def card_height(text: str, width: int) -> int:
    cols = max(12, (width - 24) // 8)
    rows = 0.0
    for line in text.split("\n"):
        n = max(1, math.ceil(dispw(rendered(line)) / cols))
        rows += n * (1.5 if line.startswith("#") else 1.0)  # 헤딩은 폰트가 크다
    return max(MIN_H, int(rows * LINE_H) + PAD_H)


def safe_name(name: str) -> str:
    out = ILLEGAL_FS.sub("-", name.replace("`", "")).strip()
    out = re.sub(r"\s+", " ", out).rstrip(". ")
    return out or "untitled"


def fence_for(code: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


# ── 파서 ──────────────────────────────────────────────────────────────
def markdown_lines(lines: list[str]) -> dict[int, str]:
    """percent 셀을 훑어 '마크다운 셀 안의 줄'만 {행번호: 본문} 으로 돌려준다."""
    out: dict[int, str] = {}
    in_md = False
    for i, raw in enumerate(lines, start=1):
        if RE_CELL.match(raw):
            in_md = "[markdown]" in raw
            continue
        if not in_md:
            continue
        if raw.startswith("# "):
            out[i] = raw[2:].rstrip()
        elif raw.rstrip() == "#":
            out[i] = ""
        else:  # 마크다운 셀이 끝나고 코드가 시작된 경우
            in_md = False
    return out


# ── 파일 입출력 추출 ──────────────────────────────────────────────────
READ_FUNCS = {"read_csv", "read_excel", "read_parquet", "read_json", "read_table",
              "read_pickle", "read_feather", "loadtxt", "genfromtxt"}
WRITE_FUNCS = {"to_csv", "to_excel", "to_parquet", "to_json", "to_pickle",
               "savefig", "savetxt", "savez", "to_feather"}
PATH_READ_METHODS = {"read_text", "read_bytes"}
PATH_WRITE_METHODS = {"write_text", "write_bytes"}


def resolve_path(node: ast.AST, consts: dict, loc: dict) -> str | None:
    """경로 표현식을 논리 경로 문자열로 푼다. 못 풀면 None(= 조용히 버리지 않고 집계한다)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return loc.get(node.id) or consts.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr == "parent":   # OUTPUT_DIR.parent
        base = resolve_path(node.value, consts, loc)
        if base is None:
            return None
        return base.rsplit("/", 1)[0] if "/" in base else "."
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        lhs = resolve_path(node.left, consts, loc)
        rhs = resolve_path(node.right, consts, loc)
        if not (lhs and rhs):
            return None
        return rhs if lhs == "." else f"{lhs}/{rhs}"
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = resolve_path(v.value, consts, loc)
                try:
                    out.append(inner if inner else "{" + ast.unparse(v.value) + "}")
                except Exception:  # pragma: no cover
                    out.append("{?}")
        return "".join(out)
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in ("Path", "str") and node.args:
            return resolve_path(node.args[0], consts, loc)
    return None


def collect_consts(tree: ast.Module, consts: dict) -> dict:
    """모듈 최상위의 경로 상수를 모은다 (PREDICTION_PATH = OUTPUT_DIR / '...' 같은 것)."""
    for n in tree.body:
        targets = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, ast.AnnAssign) and n.value else [])
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            val = resolve_path(n.value, consts, {})
            if val and ("/" in val or "." in val or t.id.endswith(("_PATH", "_DIR", "_FILE"))):
                consts[t.id] = val
    return consts


def local_paths(fn: ast.AST, consts: dict) -> dict:
    """함수 인자 기본값 + 지역 대입에서 경로 변수를 모은다."""
    loc: dict = {}
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = fn.args
        pos = a.posonlyargs + a.args
        for arg, default in zip(pos[len(pos) - len(a.defaults):], a.defaults):
            v = resolve_path(default, consts, {})
            if v:
                loc[arg.arg] = v
        for kw, default in zip(a.kwonlyargs, a.kw_defaults):
            v = resolve_path(default, consts, {}) if default is not None else None
            if v:
                loc[kw.arg] = v
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    v = resolve_path(n.value, consts, loc)
                    if v and ("/" in v or "." in v):
                        loc[t.id] = v
    return loc


def collect_loop_strs(tree: ast.AST) -> dict:
    """`for _name, _tbl in [("ch1_eda_hourly", t1), ...]` 처럼 루프가 이름을 정하는 경우를
    변수 -> 문자열 목록 으로 모은다. 이걸 안 하면 저장 호출이 조용히 누락된다."""
    out: dict = {}

    def seq(node):
        return node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else None

    for n in ast.walk(tree):
        if not isinstance(n, ast.For):
            continue
        elts = seq(n.iter)
        if not elts:
            continue
        if isinstance(n.target, ast.Name):
            vals = [e.value for e in elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(vals) == len(elts):
                out.setdefault(n.target.id, []).extend(vals)
        elif isinstance(n.target, ast.Tuple):
            rows = [seq(e) for e in elts]
            if not all(rows) or not all(len(r) == len(n.target.elts) for r in rows):
                continue
            for i, tgt in enumerate(n.target.elts):
                if not isinstance(tgt, ast.Name):
                    continue
                vals = [r[i].value for r in rows
                        if isinstance(r[i], ast.Constant) and isinstance(r[i].value, str)]
                if len(vals) == len(rows):
                    out.setdefault(tgt.id, []).extend(vals)
    return out


def extract_io(call: ast.Call, module: str, consts: dict, loc: dict, loops: dict | None = None):
    """호출 하나에서 (IOItem 목록, 미해결 표현식 목록) 을 만든다."""
    items, unresolved = [], []
    loops = loops or {}
    fn = call.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    ln = call.lineno

    def const(i):
        return (call.args[i].value if len(call.args) > i
                and isinstance(call.args[i], ast.Constant)
                and isinstance(call.args[i].value, str) else None)

    def names_at(i):
        """리터럴이면 [값] · 루프변수면 [값들] · 못 풀면 None(→ 미해결로 집계)."""
        if len(call.args) <= i:
            return None
        a = call.args[i]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return [a.value]
        if isinstance(a, ast.Name) and a.id in loops:
            return list(dict.fromkeys(loops[a.id]))
        if isinstance(a, ast.JoinedStr):     # f"{fid}_src" — 헬퍼 내부의 저장 규칙
            got = resolve_path(a, consts, loc)
            return [got] if got else None
        return None

    def miss(node):
        try:
            unresolved.append(f"{module}:{ln} {name}({ast.unparse(node)[:60]})")
        except Exception:  # pragma: no cover
            unresolved.append(f"{module}:{ln} {name}(?)")

    def add(kind, node, detail=""):
        p = resolve_path(node, consts, loc)
        if p is None:
            miss(node)
            return
        items.append(IOItem(kind, p, p.rsplit("/", 1)[-1], detail, module, ln))

    # 프로젝트 고유 헬퍼 — s00_env 의 저장 규칙을 그대로 반영한다
    if name == "save_table":
        got = names_at(1)
        if got is None:
            miss(call.args[1] if len(call.args) > 1 else call)
        for nm in got or []:
            items.append(IOItem("write", f"outputs/tables/{nm}.csv", f"{nm}.csv",
                                "표", module, ln))
    elif name == "save_fig":
        got = names_at(1)
        if got is None:
            miss(call.args[1] if len(call.args) > 1 else call)
        title = const(2) or ""
        has_src = any(k.arg == "source_table" for k in call.keywords) or len(call.args) >= 5
        for fid in got or []:
            items.append(IOItem("write", f"outputs/figures/{fid}_*.png",
                                f"{fid}.png", title, module, ln))
            if has_src:   # save_fig 은 source_table 을 받으면 소스표를 동반 저장한다
                items.append(IOItem("write", f"outputs/tables/{fid}_src.csv",
                                    f"{fid}_src.csv", f"{title} 소스표", module, ln))
    elif name in ("glob", "rglob") and isinstance(fn, ast.Attribute):
        base = resolve_path(fn.value, consts, loc)
        pat = const(0) or (resolve_path(call.args[0], consts, loc) if call.args else None)
        if base is None or pat is None:
            miss(fn.value)
        else:
            p = f"{base}/{pat}" if base != "." else pat
            items.append(IOItem("read", p, p.rsplit("/", 1)[-1], "디렉터리 스캔", module, ln))
    elif name == "exists" and isinstance(fn, ast.Attribute):
        p = resolve_path(fn.value, consts, loc)
        if p is not None:
            items.append(IOItem("read", p, p.rsplit("/", 1)[-1], "존재 확인", module, ln))
    elif (name == "read" and isinstance(fn, ast.Attribute)
          and isinstance(fn.value, ast.Name) and fn.value.id == "nbformat"):
        add("read", call.args[0], "노트북 읽기")
    elif name in PATH_READ_METHODS and isinstance(fn, ast.Attribute):
        add("read", fn.value)
    elif name in PATH_WRITE_METHODS and isinstance(fn, ast.Attribute):
        add("write", fn.value)
    elif name in READ_FUNCS and call.args:
        add("read", call.args[0])
    elif name in WRITE_FUNCS and call.args:
        add("write", call.args[0])
    elif name in ("dump", "save") and len(call.args) >= 2:      # joblib/pickle.dump(obj, path)
        add("write", call.args[1])
    elif name == "save" and len(call.args) == 1:                # Presentation.save(path)
        add("write", call.args[0])
    elif name == "load" and call.args:                          # joblib/np.load(path)
        add("read", call.args[0])
    elif name == "open" and call.args:
        mode = const(1) or "r"
        add("write" if any(c in mode for c in "wax") else "read", call.args[0])

    # 그림 id 참조 — 파일명을 안 쓰고 fid 만으로 그림을 집어 쓰는 자리(발표자료 삽입 등).
    # 실제로 생산된 fid 일 때만 엣지가 되므로 지표 이름 "F1" 같은 건 저절로 걸러진다.
    if name != "save_fig":
        for a in call.args:
            if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                    and RE_FID.fullmatch(a.value)):
                items.append(IOItem("ref", f"outputs/figures/{a.value}_*.png",
                                    f"{a.value}.png", "그림 참조", module, ln))
    return items, unresolved


def parse_module(path: Path, seed_consts: dict | None = None) -> Module:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    md = markdown_lines(lines)
    tree = ast.parse(text, filename=str(path))

    # 1) 헤더 수집
    module_title, sec_heads = "", []  # [(lineno, num, title, heading)]
    for ln in sorted(md):
        m = RE_HEADER.match(md[ln])
        if not m:
            continue
        level, body = len(m.group(1)), m.group(2)
        if level == 2 and not module_title:
            module_title = body
        elif level == 3:
            mm = RE_SEC_NUM.match(body)
            num, title = (mm.group(1), mm.group(2)) if mm else ("", body)
            sec_heads.append((ln, num, title, body))

    # 2) 섹션 경계
    sections: list[Section] = []
    for i, (ln, num, title, heading) in enumerate(sec_heads):
        end = sec_heads[i + 1][0] if i + 1 < len(sec_heads) else len(lines) + 1
        sec = Section(num=num, title=title, heading=heading, start=ln, end=end)
        for j in range(ln + 1, end):
            if j in md:
                sec.md.append(md[j])
                b = RE_BULLET.match(md[j])
                if b and b.group(1) == "목적" and not sec.purpose:
                    sec.purpose = b.group(2)
                elif b and b.group(1) == "산출물" and not sec.outputs:
                    sec.outputs = b.group(2)
        while sec.md and not sec.md[-1].strip():
            sec.md.pop()
        sections.append(sec)

    if not sections:  # 섹션이 없는 파일도 깨지지 않게
        sections = [Section(num="", title=module_title or path.stem,
                            heading=module_title or path.stem, start=1, end=len(lines) + 1)]

    def section_of(lineno: int) -> Section:
        for s in sections:
            if s.start <= lineno < s.end:
                return s
        return sections[0]

    # 3) 최상위 함수
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        try:
            sig = f"{node.name}({ast.unparse(node.args)})"
        except Exception:  # pragma: no cover - 방어적
            sig = f"{node.name}(...)"
        if node.returns is not None:
            try:
                sig += f" -> {ast.unparse(node.returns)}"
            except Exception:  # pragma: no cover
                pass
        doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        code = "\n".join(lines[start - 1:node.end_lineno])
        calls = sorted({
            c.func.id for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        })
        section_of(node.lineno).funcs.append(
            Func(node.name, sig, doc, start, node.end_lineno, code, calls)
        )

    # 4) 파일 입출력 — 함수 안이면 함수에, 밖이면 섹션에 매단다
    consts = collect_consts(tree, dict(seed_consts or {}))
    fn_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owner_by_line: dict[int, str] = {}
    for n in fn_nodes:
        for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1):
            owner_by_line[ln] = n.name
    locs = {n.name: local_paths(n, consts) for n in fn_nodes}
    loops = collect_loop_strs(tree)
    func_by_name = {f.name: f for s in sections for f in s.funcs}

    unresolved: list[str] = []
    for c in ast.walk(tree):
        if not isinstance(c, ast.Call):
            continue
        owner = owner_by_line.get(c.lineno, "")
        items, unres = extract_io(c, path.stem, consts, locs.get(owner, {}), loops)
        unresolved += unres
        for it in items:
            it.owner = owner
            if owner and owner in func_by_name:
                func_by_name[owner].io.append(it)
            else:
                it.owner = ""
                section_of(c.lineno).io.append(it)

    for s in sections:
        for i in s.all_io:   # 저장 헬퍼 내부의 쓰기는 '개별 산출물'이 아니라 저장 규칙이다
            if path.stem == "s00_env" and i.owner in ("save_fig", "save_table"):
                i.detail = ("누적 인덱스 — 모든 save_fig 호출이 한 줄씩 추가"
                            if i.path.endswith("figure_index.csv")
                            else f"{i.owner}() 의 저장 규칙")
        s.artifacts = [i.label for i in s.all_io if i.kind == "write"]

    # 5) 모듈 의존 — 어떤 이름이 넘어오는지까지 잡는다 (인메모리 중간 산출물)
    dep_names: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.ImportFrom) and n.module):
            continue
        if not re.match(r"^s\d\d_", n.module):
            continue
        names = dep_names.setdefault(n.module, [])
        for a in n.names:
            if a.name != "*" and a.name not in names:
                names.append(a.name)
    deps = sorted(dep_names)

    return Module(stem=path.stem, path=path,
                  title=module_title or path.stem, nlines=len(lines),
                  sections=sections, deps=deps, dep_names=dep_names,
                  unresolved=unresolved)


# ── 미러 노트 ─────────────────────────────────────────────────────────
def note_rel_path(notes_dir: str, mod: Module, sec: Section) -> str:
    label = f"{sec.num} {sec.title}".strip() if sec.num else sec.title
    return f"{notes_dir}/{mod.stem}/{safe_name(label)}.md"


def render_note(mod: Module, sec: Section) -> str:
    label = f"{sec.num} {sec.title}".strip() if sec.num else sec.title
    head = [
        "---",
        f"source: src/{mod.stem}.py",
        f"lines: {sec.start}-{sec.end - 1}",
        f"module: {mod.stem}",
        f'section: "{sec.num}"',
        "generated_by: tools/build_mindmap_canvas.py",
        "---",
        "",
        "> [!warning] 자동 생성 파일 — 직접 수정하지 마세요.",
        "> `python tools/build_mindmap_canvas.py` (pod 에서는 `make mindmap`) 로 재생성됩니다.",
        "",
        f"# {label}",
        "",
    ]
    body = list(sec.md)
    io = sec.all_io
    if io:
        seen, rows = set(), []
        for it in io:
            row = (it.kind, it.path, it.owner, it.lineno)
            if row in seen:
                continue
            seen.add(row)
            arrow = {"write": "📤 쓰기", "read": "📥 읽기"}.get(it.kind, "🔗 참조")
            where = f"`{it.owner}()`" if it.owner else f"L{it.lineno}"
            rows.append(f"| {arrow} | `{it.path}` | {where} | {it.detail} |")
        body += ["", "### 입출력", "",
                 "| 방향 | 경로 | 위치 | 설명 |", "|---|---|---|---|"] + rows
    parts = head + body
    for f in sec.funcs:
        fence = fence_for(f.code)
        parts += [
            "",
            "---",
            "",
            f"## {f.name}",
            "",
            f"`src/{mod.stem}.py:{f.lineno}-{f.end_lineno}` · `{f.signature}`",
            "",
            fence + "python",
            f.code,
            fence,
        ]
    return "\n".join(parts).rstrip() + "\n"


# ── 캔버스 조립 ───────────────────────────────────────────────────────
def wiki(note_path: str, alias: str, anchor: str | None = None) -> str:
    base = note_path[:-3] if note_path.endswith(".md") else note_path
    return f"[[{base}#{anchor}|{alias}]]" if anchor else f"[[{base}|{alias}]]"


def producers(modules: list[Module]) -> dict:
    """경로 -> 그 파일을 '처음 만드는' 모듈. 같은 경로를 여러 번 써도 노드는 하나다."""
    out: dict = {}
    for mod in modules:
        for it in mod.io:
            if it.kind == "write":
                out.setdefault(it.path, mod.stem)
    return out


def io_card_text(it: IOItem, icon: str, extra: str = "") -> str:
    folder, _, base = it.path.rpartition("/")
    text = f"**{icon} `{base}`**\n`{folder or '리포 루트'}`"
    if it.detail:
        text += f"\n{it.detail}"
    if "{" in base:  # 저장 헬퍼 자신의 정의 — 실제 파일명은 호출 시 결정된다
        text += "\n*(파일명은 호출 시 결정)*"
    if extra:       # 맥락 — 누가 만들었나 / 누가 다시 읽나
        text += f"\n{extra}"
    return text


def build_module(mod: Module, color: str, produced: dict, show_io: bool,
                 produced_ctx: dict | None = None, consumers_ctx: dict | None = None,
                 nav: str = ""):
    """모듈 하나의 트리를 (0,0) 기준 지역좌표로 만든다."""
    nodes: list[dict] = []
    edges: list[dict] = []
    mod_top = 0
    root_id = nid(mod.stem, "root")
    emitted: set[str] = set()

    def art_cards(items: list[IOItem]):
        """이 모듈이 '처음 만든' 산출물만 카드로 낸다 (같은 경로는 한 노드)."""
        out = []
        if not show_io:
            return out
        for it in items:
            if it.kind != "write" or produced.get(it.path) != mod.stem or it.path in emitted:
                continue
            emitted.add(it.path)
            readers = [c for c in (consumers_ctx or {}).get(it.path, []) if c != mod.stem]
            extra = f"*→ {', '.join(readers)} 가 다시 읽음*" if readers else ""
            text = io_card_text(it, "📤", extra)
            out.append((nid("art", it.path), text, card_height(text, W_IO)))
        return out

    n_sec, n_fn = len(mod.sections), len(mod.funcs)
    dep_txt = ", ".join(mod.deps) if mod.deps else "없음 (진입점)"
    num = mod.title.split(".", 1)[0].strip()
    subtitle = mod.title.split(".", 1)[1].strip() if "." in mod.title else mod.title
    n_w = len({i.path for i in mod.io if i.kind == "write"})
    n_r = len({i.path for i in mod.io if i.kind == "read"})
    root_text = (  # 사용자가 손으로 만든 카드의 표기(`# 1.diagnose.py`)를 승계한다
        f"# {num}. {mod.stem}.py\n"
        f"{subtitle}\n\n"
        f"`{mod.nlines}줄 · 섹션 {n_sec} · 함수 {n_fn}`\n"
        f"`읽기 {n_r} · 쓰기 {n_w}`\n"
        f"입력 ← {dep_txt}"
        + (f"\n\n{nav}" if nav else "")
    )
    root_h = card_height(root_text, W_ROOT)

    y = mod_top
    for sec in mod.sections:
        label = f"{sec.num} {sec.title}".strip() if sec.num else sec.title
        bits = [f"### {wiki(sec.note_path, label)}"]
        if sec.purpose:
            bits.append(f"**목적** {sec.purpose}")
        if sec.outputs:
            bits.append(f"**산출물** {sec.outputs}")
        bits.append(f"`L{sec.start}–{sec.end - 1}` · 함수 {len(sec.funcs)}")
        sec_text = "\n".join(bits)
        sec_h = card_height(sec_text, W_SEC)
        sec_id = nid(mod.stem, "sec", sec.key)

        # 행 = (함수카드, 그 함수가 만든 산출물들). 함수 밖 산출물은 마지막 행.
        rows = []
        for f in sec.funcs:
            ftext = f"**{wiki(sec.note_path, f.name + '()', anchor=f.name)}**"
            if f.doc:
                ftext += f"\n{f.doc}"
            ftext += f"\n`L{f.lineno}–{f.end_lineno}`"
            rows.append((f, ftext, card_height(ftext, W_FUNC), art_cards(f.io)))
        direct = art_cards(sec.io)
        if direct:
            rows.append((None, "", 0, direct))

        heights = []
        for _f, _t, fh, arts in rows:
            ah = sum(a[2] for a in arts) + GAP_ART * max(0, len(arts) - 1)
            heights.append(max(fh, ah))
        block_h = sum(heights) + GAP_FUNC * max(0, len(rows) - 1)
        row_h = max(sec_h, block_h)

        nodes.append({"id": sec_id, "type": "text", "text": sec_text,
                      "x": X_SEC, "y": y + (row_h - sec_h) // 2,
                      "width": W_SEC, "height": sec_h, "color": color})
        edges.append({"id": nid(mod.stem, "e", "root", sec_id), "fromNode": root_id,
                      "fromSide": "right", "toNode": sec_id, "toSide": "left"})

        fy = y + (row_h - block_h) // 2
        for (f, ftext, fh, arts), hh in zip(rows, heights):
            if f is not None:
                fn_id = nid(mod.stem, "fn", f.name)
                nodes.append({"id": fn_id, "type": "text", "text": ftext, "x": X_FUNC,
                              "y": fy + (hh - fh) // 2, "width": W_FUNC, "height": fh})
                edges.append({"id": nid(mod.stem, "e", sec_id, fn_id), "fromNode": sec_id,
                              "fromSide": "right", "toNode": fn_id, "toSide": "left"})
            ah = sum(a[2] for a in arts) + GAP_ART * max(0, len(arts) - 1)
            ay = fy + (hh - ah) // 2
            for art_id, art_text, art_h in arts:
                nodes.append({"id": art_id, "type": "text", "text": art_text, "x": X_ART,
                              "y": ay, "width": W_IO, "height": art_h, "color": COLOR_ART})
                ay += art_h + GAP_ART
            fy += hh + GAP_FUNC

        y += row_h + GAP_SEC

    block_bottom = y - GAP_SEC
    nodes.append({"id": root_id, "type": "text", "text": root_text, "x": X_ROOT,
                  "y": mod_top + max(0, (block_bottom - mod_top - root_h) // 2),
                  "width": W_ROOT, "height": root_h, "color": color})

    # 외부 입력 — 이 빌드의 어떤 모듈도 만들지 않는 파일만 노드로 세운다
    if show_io:
        seen_in: set[str] = set()
        iy = mod_top
        for it in mod.io:
            if it.kind != "read" or it.path in produced or it.path in seen_in:
                continue
            seen_in.add(it.path)
            maker = (produced_ctx or {}).get(it.path)
            extra = f"*← {maker}*" if maker else ""
            if "*" in it.path:   # glob 팬인 — 엣지 100개 대신 해당 개수로 표현한다
                hits = sum(1 for p in (produced_ctx or {}) if fnmatch.fnmatch(p, it.path))
                if hits:
                    extra = f"*← 파이프라인 산출물 {hits}개가 이 패턴에 해당*"
            text = io_card_text(it, "📥", extra)
            h = card_height(text, W_IO)
            nodes.append({"id": nid("in", mod.stem, it.path), "type": "text", "text": text,
                          "x": X_IN, "y": iy, "width": W_IO, "height": h, "color": COLOR_IN})
            iy += h + GAP_FUNC

    gx0 = min(n["x"] for n in nodes) - PAD_GROUP
    gx1 = max(n["x"] + n["width"] for n in nodes) + PAD_GROUP
    gy0 = min(n["y"] for n in nodes) - PAD_GROUP - LABEL_SPACE
    gy1 = max(n["y"] + n["height"] for n in nodes) + PAD_GROUP
    nodes.append({
        "id": nid(mod.stem, "group"), "type": "group",
        "label": f"{mod.stem}.py — {mod.title}",
        "x": gx0, "y": gy0, "width": gx1 - gx0, "height": gy1 - gy0, "color": color,
    })

    minx = min(n["x"] for n in nodes)
    miny = min(n["y"] for n in nodes)
    for n in nodes:
        n["x"] -= minx
        n["y"] -= miny
    w = max(n["x"] + n["width"] for n in nodes)
    h = max(n["y"] + n["height"] for n in nodes)
    return nodes, edges, w, h


def build_canvas(modules: list[Module], notes_dir: str, call_edges: bool,
                 cols: int, show_io: bool = True, context: list[Module] | None = None,
                 split_dir: str = ""):
    ctx = context or modules
    for mod in ctx:
        for sec in mod.sections:
            sec.note_path = note_rel_path(notes_dir, mod, sec)

    produced = producers(modules) if show_io else {}
    # 입력 카드에 붙일 "누가 만들었나" 문구. 저장 헬퍼가 쓰는 누적 파일은 모듈을 주장하지 않는다.
    produced_ctx: dict = {}
    if show_io:
        for m in ctx:
            for it in m.io:
                if it.kind == "write" and it.path not in produced_ctx:
                    produced_ctx[it.path] = (it.detail if it.detail.startswith("누적")
                                             else f"{m.stem} 가 만든 파일")
    consumers_ctx: dict = {}
    for m in ctx:
        for it in m.io:
            if it.kind == "read":
                seen = consumers_ctx.setdefault(it.path, [])
                if m.stem not in seen:
                    seen.append(m.stem)

    # 그림 id 참조는 '전체 맥락' 기준으로 판정한다. 모듈 1장만 그릴 때도
    # 다른 모듈이 만든 그림을 쓰는 사실이 사라지면 안 된다.
    if show_io:
        for m in ctx:
            for sec in m.sections:
                for it in sec.all_io:
                    if it.kind == "ref":
                        it.kind = "read" if it.path in produced_ctx else "drop"

    order = {m.stem: i for i, m in enumerate(ctx)}

    def nav_line(mod: Module) -> str:
        """모듈별 캔버스로 쪼갤 때만 — 파이프라인 앞뒤 캔버스로 건너뛰는 링크."""
        if not split_dir:
            return ""
        i = order.get(mod.stem)
        if i is None:
            return ""
        bits = []
        if i > 0:
            prev = ctx[i - 1].stem
            bits.append(f"◀ [[{split_dir}/{prev}.canvas|{prev}]]")
        if i + 1 < len(ctx):
            nxt = ctx[i + 1].stem
            bits.append(f"[[{split_dir}/{nxt}.canvas|{nxt}]] ▶")
        return " · ".join(bits)

    blocks = [build_module(m, MODULE_COLORS[order.get(m.stem, i) % len(MODULE_COLORS)],
                           produced, show_io, produced_ctx, consumers_ctx, nav_line(m))
              for i, m in enumerate(modules)]
    col_w = max(w for _, _, w, _ in blocks)

    nodes: list[dict] = []
    edges: list[dict] = []
    x_off = y_off = row_h = 0
    for i, (bn, be, _w, h) in enumerate(blocks):
        for n in bn:
            n["x"] += x_off
            n["y"] += y_off
        nodes += bn
        edges += be
        row_h = max(row_h, h)
        if cols >= 1 and (i + 1) % cols == 0:
            x_off, y_off, row_h = 0, y_off + row_h + GAP_MODULE, 0
        else:
            x_off += col_w + GAP_MODULE_X

    # 모듈 간 의존 — 트리를 피해 왼쪽으로 돌린다
    present = {m.stem for m in modules}
    for mod in modules:
        for dep in mod.deps:
            if dep not in present:
                continue
            names = mod.dep_names.get(dep, [])
            shown = ", ".join(names[:3])
            if len(names) > 3:
                shown += f" 외 {len(names) - 3}"
            edges.append({
                "id": nid("dep", dep, mod.stem),
                "fromNode": nid(dep, "root"), "fromSide": "left",
                "toNode": nid(mod.stem, "root"), "toSide": "left",
                "color": "6", "label": shown or "import",
            })

    # 데이터 흐름 — 입력 → 코드 → 산출물 → (그 산출물을 읽는 곳)
    if show_io:
        seen_io: set = set()
        for mod in modules:
            for sec in mod.sections:
                for it in sec.all_io:
                    if it.kind == "drop":
                        continue
                    pid = (nid(mod.stem, "fn", it.owner) if it.owner
                           else nid(mod.stem, "sec", sec.key))
                    if it.kind == "write":
                        key = ("w", pid, it.path)
                        if key in seen_io:
                            continue
                        seen_io.add(key)
                        edges.append({
                            "id": nid("io", *key), "fromNode": pid, "fromSide": "right",
                            "toNode": nid("art", it.path), "toSide": "left",
                            "color": COLOR_ART,
                        })
                    elif it.path in produced:      # 앞 단계가 만든 파일을 다시 읽는다
                        src = nid("art", it.path)
                        key = ("r", src, pid)
                        if key in seen_io:
                            continue
                        seen_io.add(key)
                        edges.append({
                            "id": nid("io", *key), "fromNode": src, "fromSide": "bottom",
                            "toNode": pid, "toSide": "top", "color": COLOR_FLOW,
                            "label": "그림 사용" if it.kind == "ref" else "재사용",
                        })
                    else:                          # 외부 입력 파일
                        src = nid("in", mod.stem, it.path)
                        key = ("r", src, pid)
                        if key in seen_io:
                            continue
                        seen_io.add(key)
                        edges.append({
                            "id": nid("io", *key), "fromNode": src, "fromSide": "right",
                            "toNode": pid, "toSide": "left", "color": COLOR_IN,
                        })

    # 모듈 내부 함수 호출 — 오른쪽으로 돌린다
    if call_edges:
        for mod in modules:
            names = {f.name for f in mod.funcs}
            for f in mod.funcs:
                for callee in f.calls:
                    if callee in names and callee != f.name:
                        edges.append({
                            "id": nid(mod.stem, "call", f.name, callee),
                            "fromNode": nid(mod.stem, "fn", f.name), "fromSide": "right",
                            "toNode": nid(mod.stem, "fn", callee), "toSide": "right",
                            "color": "3",
                        })

    return {"nodes": nodes, "edges": edges}


# ── 검증 (표본 아님 — 전수) ───────────────────────────────────────────
def verify(canvas: dict, modules: list[Module], notes: dict[str, str],
           show_io: bool = True, canvas_targets: set | None = None) -> list[str]:
    canvas_targets = canvas_targets or set()
    errs: list[str] = []
    nodes, edges = canvas["nodes"], canvas["edges"]

    # 1. JSON 왕복
    try:
        round_trip = json.loads(json.dumps(canvas, ensure_ascii=False))
        assert round_trip == canvas
    except Exception as e:
        errs.append(f"JSON 왕복 실패: {e}")

    # 2. id 유일성
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        errs.append(f"노드 id 중복 {len(dup)}건: {sorted(dup)[:5]}")
    eids = [e["id"] for e in edges]
    if len(eids) != len(set(eids)):
        errs.append(f"엣지 id 중복 {len(eids) - len(set(eids))}건")

    # 3. 개수 대조 — ast / 정규식 / 노드 세 경로가 모두 일치해야 한다
    idset = set(ids)
    for mod in modules:
        regex_defs = [
            m.group(1) for m in
            (RE_TOPDEF.match(l) for l in mod.path.read_text(encoding="utf-8").splitlines())
            if m
        ]
        ast_defs = [f.name for f in mod.funcs]
        if sorted(regex_defs) != sorted(ast_defs):
            errs.append(
                f"{mod.stem}: 정규식 def {len(regex_defs)}개 ≠ ast def {len(ast_defs)}개 "
                f"(차이: {sorted(set(regex_defs) ^ set(ast_defs))[:5]})"
            )
        if len(set(ast_defs)) != len(ast_defs):
            errs.append(f"{mod.stem}: 동명 최상위 함수 존재 — 노드 id 충돌")
        for name in ast_defs:
            if nid(mod.stem, "fn", name) not in idset:
                errs.append(f"{mod.stem}: 함수 노드 누락 {name}")

    produced = producers(modules) if show_io else {}
    n_art = len(produced)
    n_in = 0
    if show_io:
        for mod in modules:
            n_in += len({i.path for i in mod.io if i.kind == "read" and i.path not in produced})
    expected = sum(2 + len(m.sections) + len(m.funcs) for m in modules) + n_art + n_in
    if len(nodes) != expected:
        errs.append(f"노드 수 불일치: {len(nodes)} ≠ 기대 {expected} "
                    f"(root+group {2 * len(modules)} · 섹션 {sum(len(m.sections) for m in modules)} "
                    f"· 함수 {sum(len(m.funcs) for m in modules)} · 산출물 {n_art} · 입력 {n_in})")

    # 3b. 모든 I/O 항목이 노드로 존재하고, 고립된 I/O 노드가 없어야 한다
    if show_io:
        for mod in modules:
            for it in mod.io:
                if it.kind == "drop":
                    continue          # 생산된 그림과 안 맞는 F## 리터럴 — 노드도 엣지도 없다
                want = (nid("art", it.path) if it.kind == "write"
                        else (nid("art", it.path) if it.path in produced
                              else nid("in", mod.stem, it.path)))
                if want not in idset:
                    errs.append(f"{mod.stem}:{it.lineno} I/O 노드 누락 {it.path}")
        touched = {e["fromNode"] for e in edges} | {e["toNode"] for e in edges}
        for n in nodes:
            if n["type"] == "text" and n["text"].startswith(("**📤", "**📥"))                     and n["id"] not in touched:
                errs.append(f"고립된 I/O 노드: {n['text'].splitlines()[0]}")

    # 4. 고아 엣지
    for e in edges:
        for k in ("fromNode", "toNode"):
            if e[k] not in idset:
                errs.append(f"고아 엣지 {e['id']}: {k}={e[k]}")

    # 5. 위키링크 대상 실존
    link_re = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|[^\]]*)?\]\]")
    heads = {p: {h[3:].strip() for h in c.splitlines() if h.startswith("## ")}
             for p, c in notes.items()}
    for n in nodes:
        for target, anchor in link_re.findall(n.get("text", "")):
            if target.endswith(".canvas"):      # 모듈별 캔버스 간 이동 링크
                if target not in canvas_targets:
                    errs.append(f"캔버스 링크 대상 없음: {target}")
                continue
            path = target + ".md"
            if path not in notes:
                errs.append(f"링크 대상 없음: {path}")
            elif anchor and anchor not in heads[path]:
                errs.append(f"헤딩 없음: {path}#{anchor}")

    # 6. 겹침 / 그룹 포함 — --keep-layout 이어도 건너뛰지 않는다
    if True:
        cards = [n for n in nodes if n["type"] != "group"]
        for i in range(len(cards)):
            a = cards[i]
            for b in cards[i + 1:]:
                if (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                        and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"]):
                    errs.append(f"노드 겹침: {a['id']} ↔ {b['id']}")
        groups = [n for n in nodes if n["type"] == "group"]
        counted = {g["id"]: 0 for g in groups}
        for c in cards:  # 모든 카드는 '정확히 한 그룹' 안에 완전히 들어가야 한다
            owners = [g for g in groups
                      if g["x"] <= c["x"] and c["x"] + c["width"] <= g["x"] + g["width"]
                      and g["y"] <= c["y"] and c["y"] + c["height"] <= g["y"] + g["height"]]
            if len(owners) != 1:
                errs.append(f"카드 {c['id']} 가 속한 그룹이 {len(owners)}개 (1이어야 함)")
            for g in owners:
                counted[g["id"]] += 1
        for g in groups:
            if counted[g["id"]] == 0:
                errs.append(f"빈 그룹: {g['label']}")

    # 7. 노트 ↔ 함수 1:1
    for mod in modules:
        for sec in mod.sections:
            body = notes.get(sec.note_path)
            if body is None:
                errs.append(f"노트 누락: {sec.note_path}")
                continue
            for f in sec.funcs:
                if body.count(f"\n## {f.name}\n") != 1:
                    errs.append(f"{sec.note_path}: '## {f.name}' 헤딩이 1회가 아님")
                if f.code.rstrip() not in body:
                    errs.append(f"{sec.note_path}: {f.name} 코드 본문 누락")
    return errs


# ── 메인 ──────────────────────────────────────────────────────────────
def emit(render: list[Module], context: list[Module], out_rel: str, args) -> int:
    """캔버스 1장을 만들고 검증한 뒤 기록한다. 0=성공."""
    notes_dir = args.notes_dir.rstrip("/")
    canvas = build_canvas(render, notes_dir, not args.no_call_edges,
                          max(1, args.cols), not args.no_io, context,
                          (args.split or "").rstrip("/"))
    notes = {sec.note_path: render_note(mod, sec) for mod in render for sec in mod.sections}

    kept = 0
    out_path = ROOT / out_rel
    if args.keep_layout and out_path.exists():
        old = {n["id"]: n for n in json.loads(out_path.read_text(encoding="utf-8"))["nodes"]}
        for n in canvas["nodes"]:
            if n["id"] in old:
                n["x"], n["y"] = old[n["id"]]["x"], old[n["id"]]["y"]
                kept += 1

    sd = (args.split or "").rstrip("/")
    targets = {f"{sd}/{m.stem}.canvas" for m in context} if sd else set()
    errs = verify(canvas, render, notes, not args.no_io, targets)
    if errs and args.keep_layout:
        errs.append("--keep-layout 로 되살린 좌표가 현재 카드 크기와 충돌합니다. "
                    "옵션 없이 재생성하세요.")

    n_sec = sum(len(m.sections) for m in render)
    n_fn = sum(len(m.funcs) for m in render)
    prod = producers(render) if not args.no_io else {}
    n_art = len(prod)
    n_read = sum(1 for m in render for i in m.io if i.kind == "read")
    dropped_ref = sum(1 for m in render for i in m.io if i.kind == "drop")
    nodes, edges = len(canvas["nodes"]), len(canvas["edges"])
    print(f"{out_rel}\n  섹션 {n_sec} · 함수 {n_fn} · 산출물 {n_art} · 읽기 {n_read} "
          f"→ 노드 {nodes} · 엣지 {edges} · 노트 {len(notes)}"
          + (f" · 좌표유지 {kept}" if kept else ""))
    if dropped_ref:
        print(f"  · 생산된 그림과 안 맞는 F## 리터럴 {dropped_ref}건은 엣지로 세우지 않음")
    for u in [u for m in render for u in m.unresolved]:
        print("  ? 경로 미해결(노드 없음): " + u)

    if errs:
        print(f"  검증 실패 {len(errs)}건:", file=sys.stderr)
        for e in errs[:40]:
            print("    ✗ " + e, file=sys.stderr)
        if len(errs) > 40:
            print(f"    … 외 {len(errs) - 40}건", file=sys.stderr)
        return 1
    print("  검증 통과: id유일·개수대조(ast=regex=노드)·고아엣지0·링크실존·겹침0·노트1:1")

    if args.dry_run:
        return 0
    if not args.no_notes:
        for rel, body in notes.items():
            f = ROOT / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(body, encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(canvas, ensure_ascii=False, indent="	"), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="src → Obsidian Canvas 마인드맵")
    ap.add_argument("--modules", nargs="*", default=None, help="접두사 필터 (예: s02 s03)")
    ap.add_argument("--out", default="Call Graph.canvas")
    ap.add_argument("--split", metavar="DIR", default=None,
                    help="모듈마다 캔버스 1장씩 DIR 에 쓴다 (예: --split maps)")
    ap.add_argument("--notes-dir", default="mindmap")
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--no-call-edges", action="store_true")
    ap.add_argument("--no-io", action="store_true", help="입력·산출물 노드를 빼고 코드 구조만")
    ap.add_argument("--cols", type=int, default=1,
                    help="모듈 group 을 가로 N개씩 배치 (1=세로 한 줄)")
    ap.add_argument("--keep-layout", action="store_true", help="기존 캔버스의 노드 좌표 유지")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_paths = sorted(SRC_DIR.glob("s*.py"))
    paths = all_paths
    if args.modules:
        paths = [p for p in all_paths if any(p.stem.startswith(m) for m in args.modules)]
    if not paths:
        print("대상 모듈이 없습니다.", file=sys.stderr)
        return 2

    seed: dict = {}
    s00 = SRC_DIR / "s00_env.py"
    if s00.exists():  # 경로 상수는 전부 s00_env 에 있다 — 부분 빌드에서도 읽는다
        collect_consts(ast.parse(s00.read_text(encoding="utf-8")), seed)

    # 맥락은 항상 전체를 판다 — 모듈 1장만 그려도 "누가 만들고 누가 읽는지" 를 잃지 않는다
    context = [parse_module(p, seed) for p in all_paths]
    by_stem = {m.stem: m for m in context}
    render = [by_stem[p.stem] for p in paths]

    if args.split:
        rc = 0
        for mod in render:
            rc |= emit([mod], context, f"{args.split.rstrip('/')}/{mod.stem}.canvas", args)
        print(f"\n모듈별 캔버스 {len(render)}장 → {args.split.rstrip('/')}/")
        if args.dry_run:
            print("(dry-run — 파일을 쓰지 않았습니다)")
        return rc

    rc = emit(render, context, args.out, args)
    if args.dry_run:
        print("(dry-run — 파일을 쓰지 않았습니다)")
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
    raise SystemExit(main())
