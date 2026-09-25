"""10.5절 모델 번들 테스트 — 평가·배포 번들의 무결성 · 비트 재현 · 매니페스트 계약.

번들은 "제출 예측을 만든 바로 그 모델"을 파일로 남긴 것이다. 따라서
**다시 읽어 예측한 값이 메모리 예측과 비트 단위로 같아야** 하고, 매니페스트의
임계값·학습 구간·무결성 목록이 파이프라인 실측과 정확히 일치해야 한다.

테스트는 FAST 모드로 돌므로 번들은 `outputs/models/fast/{eval,deploy}/` 에 생긴다.
모든 테스트는 `s10` fixture 에 의존한다 — s10 import 가 번들을 새로 쓰므로,
그보다 먼저 디스크 번들을 읽으면 이전 실행의 잔여 번들을 검증하게 된다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# serving/ 이 없는 환경(노트북·데이터만 받은 경우)에서는 이 모듈을 통째로 건너뛴다
pytest.importorskip("serving")

ROLES = ("eval", "deploy")


def _sha(b: bytes) -> str:
    """바이트열의 sha256 16진 문자열."""
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def bundle_dirs(s10, project_root) -> dict:
    """역할별 번들 디렉터리 (s10 이 방금 쓴 FAST 번들)."""
    base = project_root / "outputs" / "models" / "fast"
    assert Path(s10.BUNDLE_BASE).resolve() == base.resolve(), "테스트(FAST)는 fast/ 번들만 써야 한다"
    return {r: base / r for r in ROLES}


@pytest.fixture(scope="module")
def manifests(bundle_dirs) -> dict:
    """역할별 매니페스트(dict)."""
    return {r: json.loads((d / "manifest.json").read_bytes()) for r, d in bundle_dirs.items()}


@pytest.fixture(scope="module")
def serving_eval(bundle_dirs):
    """서빙 로더로 연 평가 번들 (FAST 허용)."""
    import serving

    return serving.load_bundle(bundle_dirs["eval"], allow_fast=True)


# ══════════════════════════════════════════════════════════════════════
# 1. 파일 무결성
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("role", ROLES)
def test_bundle_files_exist_and_sha_match(role, bundle_dirs, manifests):
    """매니페스트·목록의 모든 파일이 존재하고 sha256 이 일치한다."""
    d, man = bundle_dirs[role], manifests[role]
    assert (d / "manifest.json").is_file()
    assert len(man["files_sha256"]) > 0
    for name, sha in man["files_sha256"].items():
        p = d / name
        assert p.is_file(), f"번들 파일 없음: {name}"
        assert _sha(p.read_bytes()) == sha, f"sha256 불일치: {name}"


@pytest.mark.parametrize("role", ROLES)
def test_bundle_dir_has_no_stray_files(role, bundle_dirs, manifests):
    """번들 디렉터리에는 목록의 파일과 매니페스트만 있다(이전 실행 잔여 없음)."""
    on_disk = {p.name for p in bundle_dirs[role].iterdir()}
    assert on_disk == set(manifests[role]["files_sha256"]) | {"manifest.json"}


@pytest.mark.parametrize("role", ROLES)
def test_manifest_header(role, manifests):
    """스키마 버전 1 · FAST 실행 표시 · 역할 값 · 번들 id 접두어."""
    man = manifests[role]
    assert man["schema_version"] == 1
    assert man["fast_mode"] is True
    assert man["role"] == role
    assert man["bundle_id"].startswith(f"{role}-fast-")
    assert man["model_name"] == "2단계 레짐(3분류)"


def test_manifest_components_are_listed_in_sha(manifests):
    """매니페스트가 참조하는 구성 파일이 전부 무결성 목록에 있다."""
    for man in manifests.values():
        comp = man["components"]
        refs = [comp["gate"]["file"], comp["peak_classifier"]["file"], comp["calibrator"], comp["golden"]]
        refs += [f for d in comp["regressors"].values() for f in d.values()]
        refs += list(comp["fallbacks"].values()) + list(comp["quantiles"].values())
        assert set(refs) <= set(man["files_sha256"])


# ══════════════════════════════════════════════════════════════════════
# 2. 노트북 점검 · 최종 게이트
# ══════════════════════════════════════════════════════════════════════
def test_bundle_checks_all_true(s10):
    """10.5절 점검(BUNDLE_CHECKS)이 전부 True 다."""
    expected = {"eval_bitwise", "submission_match", "refit_sha", "deploy_reload", "integrity", "manifest_clean"}
    assert expected <= set(s10.BUNDLE_CHECKS)
    failed = [k for k, v in s10.BUNDLE_CHECKS.items() if v is not True]
    assert not failed, f"번들 점검 실패: {failed}"


def test_final_gate_includes_bundle_items(s10):
    """최종 게이트에 번들 4항목이 있고 모두 통과한다."""
    g = s10.gate_final
    rows = g[g["점검"].str.contains("번들")]
    assert len(rows) == 4, rows["점검"].tolist()
    assert bool(rows["통과"].all()), rows[~rows["통과"]]["점검"].tolist()


# ══════════════════════════════════════════════════════════════════════
# 3. 평가 번들 재로드 예측 == 메모리 예측 (비트)
# ══════════════════════════════════════════════════════════════════════
def test_eval_bundle_predict_bitwise(s10, serving_eval):
    """서빙 로더로 연 평가 번들의 테스트 예측이 5장 메모리 예측과 비트 단위로 같다."""
    te_s = s10.test_results["2단계 레짐(3분류)"]
    te_c = s10.test_results["피크 직접분류"]
    X = s10.feat.loc[te_s["index"], s10.FEATURE_COLS]
    got = serving_eval.predict_features(X)
    assert len(got["y_avg"]) == 336
    assert np.array_equal(got["y_avg"], te_s["pred_avg"])
    assert np.array_equal(got["y_peak"], te_s["pred_peak"])
    assert np.array_equal(got["prob"], te_c["prob"])
    assert np.array_equal(got["q10"], s10._unc["lo"])
    assert np.array_equal(got["q50"], s10._unc["mid"])
    assert np.array_equal(got["q90"], s10._unc["hi"])


def test_eval_bundle_predict_ignores_column_order(s10, serving_eval):
    """컬럼 순서를 뒤집어 넣어도 예측이 같다(예측 직전 재정렬)."""
    te_s = s10.test_results["2단계 레짐(3분류)"]
    X = s10.feat.loc[te_s["index"], s10.FEATURE_COLS]
    got = serving_eval.predict_features(X[list(reversed(s10.FEATURE_COLS))])
    ref = serving_eval.predict_features(X)
    assert all(np.array_equal(got[k], ref[k]) for k in ref)


# ══════════════════════════════════════════════════════════════════════
# 4. 임계값 — 반올림 없이 정확 일치
# ══════════════════════════════════════════════════════════════════════
def test_manifest_tau_exact(s10, manifests):
    """매니페스트 τ 가 5장 CV τ 와 정확히 같다(반올림 아님) · fold τ 4개."""
    cv = s10.cv_results["2단계 레짐(3분류)"]
    for man in manifests.values():
        tau = man["thresholds"]["tau"]
        assert tau["value"] == cv["tau"]
        assert len(tau["fold_taus"]) == 4
        assert tau["fold_taus"] == [float(x) for x in cv["fold_metrics"]["τ"]]
        assert float(np.median(tau["fold_taus"])) == tau["value"]
    assert manifests["eval"]["thresholds"]["tau_cls"]["value"] == s10.BUNDLE_SUMMARY["tau_cls"]


def test_manifest_theta_and_features(s10, manifests):
    """피처 계약 — θ·피처 순서·안전 lag 가 파이프라인과 같다."""
    for man in manifests.values():
        fc = man["feature_contract"]
        assert fc["theta"] == float(s10.THETA)
        assert fc["columns"] == list(s10.FEATURE_COLS)
        assert len(fc["columns"]) == 44
        assert fc["safe_lags"] == [24, 48, 168]


# ══════════════════════════════════════════════════════════════════════
# 5. 배포 번들 — 전 구간 재학습 · CV 산출물 상속
# ══════════════════════════════════════════════════════════════════════
def test_eval_bundle_training_window(s10, manifests):
    """평가 번들은 테스트 이전(~08-31)만 학습했고 보고서 수치의 근거다."""
    tr = manifests["eval"]["training"]
    assert pd.Timestamp(tr["last"]) < s10.TEST_START
    assert 9 not in tr["months"]
    assert tr["used_in_report"] is True
    assert manifests["eval"]["thresholds"]["inherited_from"] is None


def test_deploy_bundle_training_and_inheritance(manifests):
    """배포 번들은 09-14 23시까지 학습했고, 임계값은 평가 번들에서 상속한다."""
    ev, dp = manifests["eval"], manifests["deploy"]
    assert dp["training"]["n_rows"] > ev["training"]["n_rows"]
    assert dp["training"]["last"] == "2021-09-14 23:00:00"
    assert 9 in dp["training"]["months"]
    assert dp["thresholds"]["inherited_from"] == ev["bundle_id"]
    assert dp["intervals"]["residual"]["inherited_from"] == ev["bundle_id"]
    assert dp["training"]["used_in_report"] is False
    assert dp["submission"] is None
    # 상속한 값 자체는 평가 번들과 같다
    assert dp["thresholds"]["tau"] == ev["thresholds"]["tau"]
    assert dp["thresholds"]["tau_cls"] == ev["thresholds"]["tau_cls"]
    assert dp["files_sha256"]["calibrator_isotonic.json"] == ev["files_sha256"]["calibrator_isotonic.json"]


def test_eval_submission_binding(s10, manifests, project_root):
    """평가 번들이 제출 파일의 sha256 에 묶여 있다."""
    sub = manifests["eval"]["submission"]
    assert sub["file"] == "outputs/predictions_test_336h.csv"
    assert sub["sha256"] == _sha((project_root / sub["file"]).read_bytes())
    assert sub["bound"] is (s10.FINAL_MODEL_NAME == s10.SERVICE_MODEL_NAME)


# ══════════════════════════════════════════════════════════════════════
# 6. 블라인드 평가 — 경로·계정명 없음
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("role", ROLES)
def test_bundle_json_has_no_paths_or_account(role, bundle_dirs):
    """번들 JSON(manifest·golden·calibrator)에 절대경로·계정명이 없다."""
    d = bundle_dirs[role]
    names = ["manifest.json", "golden.json", "calibrator_isotonic.json"]
    forbidden = [t for t in (Path.home().name, ":\\", "/home/", "root") if t]
    for n in names:
        text = (d / n).read_bytes().decode("utf-8")
        hits = [t for t in forbidden if t in text]
        assert not hits, f"{role}/{n} 에 금지 토큰: {hits}"


# ══════════════════════════════════════════════════════════════════════
# 7. 산출표
# ══════════════════════════════════════════════════════════════════════
def test_bundle_table_rows(s10, project_root, manifests):
    """ch6_model_bundles.csv — 번들마다 (목록 파일 수 + 매니페스트 1) 행."""
    path = project_root / "outputs" / "tables" / "ch6_model_bundles.csv"
    assert path.is_file()
    tbl = pd.read_csv(path, encoding="utf-8-sig")
    for role, man in manifests.items():
        rows = tbl[tbl["번들"] == role]
        assert len(rows) == len(man["files_sha256"]) + 1
        assert set(rows["bundle_id"]) == {man["bundle_id"]}
        assert "manifest.json" in set(rows["파일"])
        listed = rows[rows["파일"] != "manifest.json"].set_index("파일")["sha256"].to_dict()
        assert listed == man["files_sha256"]


# ══════════════════════════════════════════════════════════════════════
# 8. 재학습 결정성
# ══════════════════════════════════════════════════════════════════════
def test_refit_eval_is_byte_identical(s10, manifests):
    """같은 cutoff 로 다시 적합하면 LightGBM 텍스트가 바이트까지 같다."""
    art = s10.fit_service_artifacts(s10.TEST_START)
    got = {n: _sha(b) for n, b in s10.component_files(art).items()}
    want = {n: s for n, s in manifests["eval"]["files_sha256"].items() if n.endswith(".lgb")}
    assert got == want
