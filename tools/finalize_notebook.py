"""실행된 노트북을 제출용으로 마감한다 — stderr 정리 + 개인정보 후스캔.

왜 별도 단계인가
----------------
노트북 안(10.6절)의 개인정보 스캔은 **자기 실행 출력을 볼 수 없다.**
`nbconvert --execute --inplace` 는 모든 셀이 끝난 **뒤에** 파일을 쓰므로,
스캔 셀이 읽는 파일은 아직 출력이 비어 있는 빌드 직후 버전이다.

실제로 이 한계 때문에 다음이 남았다.

    C:\\Users\\<사용자명>\\AppData\\Local\\Programs\\Python\\Python311\\...

서드파티 경고가 **stderr** 로 내보내는 메시지에 설치 경로가 박혀 있고,
그 경로에는 사용자명이 들어간다. 블라인드 평가에서는 위반이다.

그래서 실행이 끝난 뒤 이 스크립트로
① stderr 스트림 출력 제거 → ② 잔여 경로·사용자명 치환 → ③ 후스캔 을 수행한다.

**정상성 보장**: 노트북의 검증 게이트는 실패 시 예외를 던지므로
`output_type == "error"` 로 남는다. stderr 경고만 지우는 것은
오류를 감추지 않는다(오류가 있으면 아래 검사에서 잡아낸다).

**게이트 강제**: 최종 게이트(`outputs/tables/gate6_final.csv`)는 노트북 안에서 예외를
던지지 않고 표로만 남는다. 그래서 여기서 `통과` 가 전부 True 인지 다시 확인한다.

사용법
------
    python tools/finalize_notebook.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "자원최적화_제안모델.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

USER = Path.home().name
REDACTIONS = [
    # 사용자명이 포함된 절대경로를 통째로 가린다
    (re.compile(r"[A-Za-z]:\\+(?:Users|users)\\+[^\\\s\"']+", re.IGNORECASE), r"<경로 생략>"),
    (re.compile(r"/(?:home|Users)/[^/\s\"']+"), r"<경로 생략>"),
    (re.compile(re.escape(USER)), "<사용자>"),
]
SCAN_PATTERNS = {
    "사용자명": re.escape(USER),
    "Windows 절대경로": r"[Cc]:\\+[Uu]sers",
    "홈 경로": r"/(?:home|Users)/[A-Za-z0-9_가-힣]+",
    "이메일": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}


def redact(text: str) -> str:
    """텍스트에서 경로·사용자명을 치환한다."""
    for pat, repl in REDACTIONS:
        text = pat.sub(repl, text)
    return text


def clean_notebook(path: Path) -> dict:
    """stderr 출력을 제거하고 잔여 경로를 치환한다."""
    nb = nbformat.read(path, as_version=4)
    n_stderr = n_redacted = n_error = 0

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                n_error += 1
                kept.append(out)
                continue
            # 서드파티 경고는 stderr 로 나오며 설치 경로를 포함한다 → 제거
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                n_stderr += 1
                continue
            # stdout·표시데이터에 남은 경로는 치환한다
            if "text" in out:
                before = out["text"]
                out["text"] = redact(before)
                if out["text"] != before:
                    n_redacted += 1
            data = out.get("data")
            if isinstance(data, dict) and "text/plain" in data:
                before = data["text/plain"]
                data["text/plain"] = redact(before)
                if data["text/plain"] != before:
                    n_redacted += 1
            kept.append(out)
        cell["outputs"] = kept

    nbformat.write(nb, path)
    return {"stderr 제거": n_stderr, "경로 치환": n_redacted, "오류 출력": n_error}


def scan(path: Path) -> list[dict]:
    """마감된 노트북과 산출물에 개인정보가 남았는지 후스캔한다."""
    hits = []

    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        blobs = [cell.get("source", "") or ""]
        for out in cell.get("outputs", []) or []:
            blobs.append(str(out.get("text", "")))
            data = out.get("data")
            if isinstance(data, dict):
                blobs.append(str(data.get("text/plain", "")))
            blobs.append("".join(out.get("traceback", []) or []))
        for blob in blobs:
            for label, pat in SCAN_PATTERNS.items():
                if re.search(pat, blob):
                    hits.append({"파일": path.name, "셀": i, "유형": label})

    # .json = 모델 번들 매니페스트·자가검증 사례. 모델 텍스트(.lgb)는 경로·계정명이 없어 제외한다.
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.suffix.lower() not in (".md", ".csv", ".txt", ".json"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for label, pat in SCAN_PATTERNS.items():
            if re.search(pat, text):
                hits.append({"파일": str(f.relative_to(PROJECT_ROOT)), "셀": "-", "유형": label})
    return hits


def check_no_errors(path: Path) -> int:
    """실행 오류 출력이 남아 있는지 확인한다 (게이트 실패 여부)."""
    nb = nbformat.read(path, as_version=4)
    return sum(
        1
        for c in nb.cells
        for o in (c.get("outputs") or [])
        if o.get("output_type") == "error"
    )


def check_final_gate() -> list[str]:
    """최종 게이트 표에서 미통과 항목을 돌려준다(표가 없으면 그 자체가 실패)."""
    import csv

    path = OUTPUT_DIR / "tables" / "gate6_final.csv"
    if not path.exists():
        return [f"{path.name} 없음"]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r["점검"] for r in rows if str(r.get("통과")).strip() != "True"]


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"[FAIL] 노트북이 없다: {NOTEBOOK.name}", file=sys.stderr)
        return 1

    stats = clean_notebook(NOTEBOOK)
    print("── 노트북 마감 ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n_err = check_no_errors(NOTEBOOK)
    if n_err:
        print(f"[FAIL] 실행 오류 출력 {n_err}건이 남아 있다 — 게이트 실패", file=sys.stderr)
        return 1
    print("  실행 오류: 0건")

    failed_gate = check_final_gate()
    if failed_gate:
        print(f"[FAIL] 최종 게이트 미통과 {len(failed_gate)}건:", file=sys.stderr)
        for item in failed_gate:
            print(f"   - {item}", file=sys.stderr)
        return 1
    print("  최종 게이트: 전 항목 통과")

    hits = scan(NOTEBOOK)
    print(f"\n── 제출 전 개인정보 후스캔: {len(hits)}건 ──")
    if hits:
        for h in hits[:20]:
            print(f"  ❌ {h['파일']} (셀 {h['셀']}) — {h['유형']}")
        print("\n[FAIL] 개인정보가 남아 있다 — 제출 중단", file=sys.stderr)
        return 1
    print("  ✅ 0건 — 블라인드 평가 규정 충족")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
