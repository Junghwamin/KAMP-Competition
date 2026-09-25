"""서빙 REST API 테스트 — FastAPI 앱의 응답 계약 · 입력 검증 · 인증 · 기동 거부.

- 정상 요청(ASCII 별칭 JSON)의 24행 응답이 `Bundle.predict_day` 와 값까지 같다
- 비유한값(NaN/Infinity)·모르는 필드·계획의 전력 필드는 422 `SCHEMA` (500 이 아니다)
- 입력 계약 위반은 422 + 코드(`CALENDAR_NOT_COVERED`·`HISTORY_INVALID`)
- API 키 · 본문 크기 제한 · FAST 번들/번들 미지정 기동 거부

fastapi 가 없으면 건너뛴다. 번들은 `s10` fixture 가 방금 쓴 FAST 평가 번들을 쓴다.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# serving/ 이 없는 환경(노트북·데이터만 받은 경우)에서는 이 모듈을 통째로 건너뛴다
pytest.importorskip("serving")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

TARGET = dt.date(2021, 9, 14)
URL = "/v1/predict/day-ahead"


@pytest.fixture(scope="module")
def eval_dir(s10, project_root) -> Path:
    """FAST 평가 번들 디렉터리(s10 이 방금 썼다)."""
    return project_root / "outputs" / "models" / "fast" / "eval"


@pytest.fixture(scope="module")
def app(eval_dir):
    """API 키 없는 앱 (환경변수 KAMP_API_KEY 영향 제거)."""
    from serving.api import create_app

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("KAMP_API_KEY", raising=False)
        return create_app(str(eval_dir), allow_fast=True)


@pytest.fixture(scope="module")
def client(app) -> TestClient:
    """서버 예외를 응답(500)으로 받는 테스트 클라이언트."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def frames(s10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 한글 컬럼의 이력(08-24~09-13, 21일)·계획(09-14)."""
    from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS

    raw = s10.df_raw
    ymd = raw["날짜"].astype(int)
    hist = raw.loc[(ymd >= 20210824) & (ymd <= 20210913), HISTORY_COLUMNS].reset_index(drop=True)
    plan = raw.loc[ymd == 20210914, PLAN_COLUMNS].reset_index(drop=True)
    return hist, plan


def _rows(df: pd.DataFrame, aliases: dict) -> list[dict]:
    """한글 컬럼 프레임 → ASCII 별칭 JSON 행 (NaN → None, 날짜·시간은 int)."""
    rev = {v: k for k, v in aliases.items()}
    out = []
    for rec in df.to_dict("records"):
        row = {}
        for kor, v in rec.items():
            key = rev[kor]
            if key in ("date", "hour"):
                row[key] = int(v)
            else:
                row[key] = None if pd.isna(v) else float(v)
        out.append(row)
    return out


@pytest.fixture(scope="module")
def payload(frames) -> dict:
    """정상 요청 본문."""
    from serving.contract import HISTORY_ALIASES, PLAN_ALIASES

    hist, plan = frames
    return {
        "target_date": TARGET.isoformat(),
        "history": _rows(hist, HISTORY_ALIASES),
        "plan": _rows(plan, PLAN_ALIASES),
    }


def _post_raw(client: TestClient, body: str, headers: dict | None = None):
    """문자열 JSON 본문을 그대로 보낸다(NaN/Infinity 리터럴 포함 가능)."""
    return client.post(URL, content=body.encode("utf-8"),
                       headers={"content-type": "application/json", **(headers or {})})


def _no_leak(text: str) -> list[str]:
    """응답 텍스트에 섞인 절대경로·계정명 토큰."""
    tokens = [Path.home().name, ":\\", ":/", "/home/", "\\Users", "/Users/"]
    return [t for t in tokens if t and t in text]


# ══════════════════════════════════════════════════════════════════════
# 1. 조회 엔드포인트
# ══════════════════════════════════════════════════════════════════════
def test_healthz(client, app):
    """/healthz — 200, 번들 id 일치, 요청 id 헤더."""
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["bundle_id"] == app.state.bundle.bundle_id
    assert body["role"] == "eval"
    assert r.headers.get("X-Request-ID")


def test_model_info_contract_and_no_paths(client, app):
    """/v1/model — τ·피처 44개가 매니페스트와 같고, 경로·계정명이 없다."""
    r = client.get("/v1/model")
    assert r.status_code == 200
    body = r.json()
    man = app.state.bundle.manifest
    assert body["tau"] == man["thresholds"]["tau"]["value"]
    assert body["tau_cls"] == man["thresholds"]["tau_cls"]["value"]
    assert body["feature_columns"] == man["feature_contract"]["columns"]
    assert len(body["feature_columns"]) == 44
    assert body["bundle_id"] == man["bundle_id"]
    assert _no_leak(r.text) == []
    assert _no_leak(json.dumps(body, ensure_ascii=False)) == []


def test_docs_are_closed(client):
    """운영 앱은 스키마 문서 페이지를 열지 않는다."""
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


# ══════════════════════════════════════════════════════════════════════
# 2. 정상 예측
# ══════════════════════════════════════════════════════════════════════
def test_predict_day_ahead_matches_bundle(client, app, payload, frames):
    """09-14 예측 — 200, 24행, 값이 Bundle.predict_day 와 같고 JSON 이 엄격 직렬화된다."""
    from serving.pipeline import OUTPUT_COLUMNS

    r = client.post(URL, json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    json.dumps(body, allow_nan=False)   # NaN/inf 가 있으면 ValueError
    assert body["meta"]["target_date"] == "2021-09-14"
    assert body["meta"]["bundle_id"] == app.state.bundle.bundle_id
    assert body["meta"]["history_days"] == 21
    got = pd.DataFrame(body["rows"])
    assert len(got) == 24
    assert list(got.columns) == OUTPUT_COLUMNS

    hist, plan = frames
    want, _ = app.state.bundle.predict_day(hist, plan, target_date=TARGET, max_days=62)
    assert list(got["ts"]) == list(want["ts"])
    for c in OUTPUT_COLUMNS[1:]:
        assert np.array_equal(got[c].to_numpy(dtype=float), want[c].to_numpy(dtype=float)), c


# ══════════════════════════════════════════════════════════════════════
# 3. 스키마 오류 → 422 SCHEMA (500 아님, 입력값을 되돌려 주지 않음)
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_nonfinite_literal_is_422(client, payload, literal):
    """본문의 비유한 리터럴('"temp": Infinity' 등)은 422 SCHEMA 이고 되돌려 주지 않는다."""
    body = json.dumps(payload)
    first = json.dumps(payload["plan"][0]["temp"])
    body = body.replace(f'"temp": {first}', f'"temp": {literal}', 1)
    assert f'"temp": {literal}' in body
    r = _post_raw(client, body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "SCHEMA"
    assert "Infinity" not in r.text
    assert "NaN" not in r.text


def test_nonfinite_in_history_is_422(client, payload):
    """이력 전력값의 Infinity 도 422 SCHEMA 다."""
    body = json.dumps(payload).replace(
        f'"avg": {json.dumps(payload["history"][0]["avg"])}', '"avg": Infinity', 1)
    assert '"avg": Infinity' in body
    r = _post_raw(client, body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SCHEMA"
    assert "Infinity" not in r.text


@pytest.mark.parametrize("where", ["top", "history", "plan"])
def test_unknown_field_is_422(client, payload, where):
    """모르는 필드는 거부한다(extra='forbid')."""
    p = copy.deepcopy(payload)
    if where == "top":
        p["note"] = "x"
    else:
        p[where][0]["unknown_field"] = 1.0
    r = client.post(URL, json=p)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SCHEMA"


def test_power_field_in_plan_is_422(client, payload):
    """계획에 전력 필드(p15)가 있으면 422 — 대상일 전력 누수 차단."""
    p = copy.deepcopy(payload)
    p["plan"][0]["p15"] = 100.0
    r = client.post(URL, json=p)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SCHEMA"


def test_malformed_json_is_422_not_500(client):
    """깨진 JSON 도 422 로 답한다."""
    r = _post_raw(client, '{"target_date": "2021-09-14", "history": [')
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SCHEMA"


# ══════════════════════════════════════════════════════════════════════
# 4. 입력 계약 오류 → 422 + 코드
# ══════════════════════════════════════════════════════════════════════
def test_uncovered_calendar_is_422(client, payload):
    """2026 대상일은 공휴일표 커버리지 밖 — 422 CALENDAR_NOT_COVERED."""
    p = copy.deepcopy(payload)
    for part in ("history", "plan"):
        for row in p[part]:
            row["date"] += 50000        # YYYYMMDD 에서 5년 뒤(같은 월·일)
    p["target_date"] = "2026-09-14"
    r = client.post(URL, json=p)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "CALENDAR_NOT_COVERED"


def test_missing_history_hour_is_422(client, payload):
    """이력 1행이 빠지면 422 HISTORY_INVALID."""
    p = copy.deepcopy(payload)
    del p["history"][100]
    r = client.post(URL, json=p)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "HISTORY_INVALID"


def test_target_date_mismatch_is_422(client, payload):
    """target_date 와 계획 날짜가 다르면 422 PLAN_INVALID."""
    p = copy.deepcopy(payload)
    p["target_date"] = "2021-09-13"
    r = client.post(URL, json=p)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "PLAN_INVALID"


# ══════════════════════════════════════════════════════════════════════
# 5. API 키
# ══════════════════════════════════════════════════════════════════════
def test_api_key_required_on_v1(eval_dir):
    """api_key 설정 시 /v1/* 는 키가 있어야 하고 /healthz 는 키 없이 열린다."""
    from serving.api import create_app

    c = TestClient(create_app(str(eval_dir), allow_fast=True, api_key="k"), raise_server_exceptions=False)
    r = c.get("/v1/model")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"
    assert c.get("/v1/model", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/model", headers={"X-API-Key": "k"}).status_code == 200
    assert c.get("/healthz").status_code == 200


# ══════════════════════════════════════════════════════════════════════
# 6. 본문 크기 제한
# ══════════════════════════════════════════════════════════════════════
def test_body_too_large_is_413(client):
    """본문이 2,000,000 바이트를 넘으면(2.1MB) 파싱 전에 413."""
    r = client.post(URL, content=b"x" * 2_100_000, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "BODY_TOO_LARGE"


def test_chunked_body_is_411(client):
    """Content-Length 없는 chunked 본문은 크기 제한을 우회할 수 있으므로 411 로 거부한다."""
    def chunks():
        for _ in range(21):
            yield b" " * 100_000     # 2.1MB 공백(유효한 JSON 앞 공백)
        yield b"{}"

    r = client.post(URL, content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 411
    assert r.json()["error"]["code"] == "LENGTH_REQUIRED"


def test_non_ascii_api_key_is_401(eval_dir):
    """비ASCII 바이트가 든 X-API-Key 는 500 이 아니라 401 이어야 한다."""
    from serving.api import create_app

    c = TestClient(create_app(str(eval_dir), allow_fast=True, api_key="k"), raise_server_exceptions=False)
    r = c.get("/v1/model", headers={b"X-API-Key": b"\xff"})
    assert r.status_code == 401


# ══════════════════════════════════════════════════════════════════════
# 7. 기동 거부
# ══════════════════════════════════════════════════════════════════════
def test_fast_bundle_refused_without_flag(eval_dir):
    """allow_fast 없이 FAST 번들로는 앱이 뜨지 않는다(BundleError FAST_BUNDLE)."""
    from serving.api import create_app
    from serving.errors import BundleError

    with pytest.raises(BundleError) as ei:
        create_app(str(eval_dir))
    assert ei.value.code == "FAST_BUNDLE"
    assert ei.value.exit_code == 3


def test_missing_bundle_dir_refused(monkeypatch):
    """번들 경로도 KAMP_BUNDLE_DIR 도 없으면 BundleError."""
    from serving.api import create_app
    from serving.errors import BundleError

    monkeypatch.delenv("KAMP_BUNDLE_DIR", raising=False)
    with pytest.raises(BundleError) as ei:
        create_app()
    assert ei.value.code == "BUNDLE_NOT_FOUND"


def test_nonexistent_bundle_dir_refused(tmp_path):
    """매니페스트가 없는 디렉터리는 BUNDLE_NOT_FOUND — 메시지에 절대경로가 없다."""
    from serving.api import create_app
    from serving.errors import BundleError

    with pytest.raises(BundleError) as ei:
        create_app(str(tmp_path), allow_fast=True)
    assert ei.value.code == "BUNDLE_NOT_FOUND"
    assert str(tmp_path) not in ei.value.message



# ══════════════════════════════════════════════════════════════════════
# 리뷰 반영 — 오류 봉투 · 요청 id · 이력 길이 오류 코드
# ══════════════════════════════════════════════════════════════════════
def test_unknown_path_uses_error_envelope(client):
    """404 도 같은 오류 봉투({"error": {...}})로 돌려준다."""
    r = client.get("/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_unhandled_error_has_request_id(eval_dir, payload, monkeypatch):
    """처리되지 않은 예외도 500 INTERNAL + X-Request-ID 헤더(추적 가능)."""
    from serving.api import create_app

    app = create_app(str(eval_dir), allow_fast=True)
    monkeypatch.setattr(app.state.bundle, "predict_day", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post(URL, json=payload)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL"
    assert r.headers.get("X-Request-ID")


def test_short_history_gives_specific_code(client, payload):
    """7일 이력은 스키마 오류가 아니라 HISTORY_TOO_SHORT(+ required_start)로 알려준다."""
    body = dict(payload)
    body["history"] = payload["history"][-24 * 7:]
    r = client.post(URL, json=body)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "HISTORY_TOO_SHORT"
    assert "required_start" in err["detail"]


def test_nonexistent_date_is_422(client, payload):
    """존재하지 않는 날짜(20210931)는 500 이 아니라 422."""
    body = json.loads(json.dumps(payload))
    for row in body["plan"]:
        row["date"] = 20210931
    r = client.post(URL, json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "PLAN_SCHEMA"
