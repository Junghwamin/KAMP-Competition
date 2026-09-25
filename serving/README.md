# serving — 모델 번들 서빙 (운영 런북)

노트북 **10.5절**이 저장한 모델 번들을 검증하며 읽어, 매일 24:00 에 **익일 24시간**
전력(평균·peak15)과 피크 위험을 예측한다. 학습 코드는 두지 않는다 — 학습은 노트북만 한다.

> 이 폴더는 제출 zip·포드 제출 패키지에 **포함**한다(추론 전용 — 학습 코드는 없다).
> `outputs/models/` 는 제출 zip 과 git 모두에서 빠진다 — 새로 받은 환경에서는 `bash run.sh run`(FULL)으로 먼저 만든다.

---

## 1. 번들

노트북을 실행하면 `outputs/models/{full|fast}/{eval|deploy}/` 에 번들 두 개가 생긴다.

| 번들 | 학습 구간 | 용도 |
|---|---|---|
| `full/eval` | 2021-01-08 ~ 08-31 | 제출 예측을 만든 **바로 그 모델**. 재로드 예측이 제출 파일과 비트 단위로 같다 |
| `full/deploy` | 2021-01-08 ~ 09-14 | **배포용**. 같은 하이퍼파라미터로 전 구간 재학습. τ·보정기·구간 반폭은 eval 에서 상속 |
| `fast/*` | (테스트 실행 산출물) | 축소 학습 — 서빙 로더가 **거부**한다 |

| 파일 | 내용 |
|---|---|
| `manifest.json` | 피처 계약(44컬럼 순서·θ·휴일)·임계값(τ, τ_cls)·구간 반폭·런타임 버전·파일별 sha256·**매니페스트 자신의 sha256**(`bundle_id` 끝 12자리) |
| `gate.lgb` · `reg_y_*_r{0,1,2}.lgb` · `fallback_y_*.lgb` | 최종모델(2단계 레짐 3분류) — LightGBM 네이티브 텍스트 |
| `peak_clf.lgb` | 피크 직접분류기 |
| `interval_q{10,50,90}.lgb` | 분위회귀(미보정 — 참고용) |
| `calibrator_isotonic.json` | 피크 확률 보정 임계점 (`np.interp` 로 적용) |
| `golden.json` | 기동 자가검증 사례(원시 이력 21일 + 계획 + 기대 피처 + 기대 출력) |

pickle 은 쓰지 않는다. 로더는 파일을 **바이트로 읽어 sha256 을 대조한 뒤** `Booster(model_str=...)`
로 연다(파일 경로로 여는 LightGBM API 는 한글이 든 절대경로를 열지 못한다).

## 2. 설치

```bash
pip install -r serving/requirements.txt
pip check          # 서빙 의존성 충돌이 없어야 한다
```

lightgbm·numpy·pandas 는 **학습 때와 같은 버전**이어야 로더가 기동한다(매니페스트 `runtime`).
다르면 `VERSION_MISMATCH` 로 거부한다. 불가피하면 `KAMP_ALLOW_VERSION_MISMATCH=1` 로 우회하되,
그래도 golden 자가검증은 통과해야 한다.

## 3. 배포 — 번들은 outputs 밖으로 복사한다

```bash
mkdir -p /srv/kamp/bundles
cp -r outputs/models/full/deploy /srv/kamp/bundles/$(python -c "import json;print(json.load(open('outputs/models/full/deploy/manifest.json',encoding='utf-8'))['bundle_id'])")
```

`make clean` 이나 노트북 재실행은 `outputs/models/` 를 지우고 다시 쓴다. 서빙 중인 번들이
바뀌지 않도록 **불변 사본**을 가리킨다. 롤백 = 이전 사본 경로로 바꾸고 재시작.

## 4. 배치 예측 (매일 24:00)

```bash
python -X utf8 -m serving verify  --bundle /srv/kamp/bundles/<id>
python -X utf8 -m serving predict --bundle /srv/kamp/bundles/<id> \
       --history history.csv --plan plan.csv --out pred_YYYYMMDD.csv
```

**입력** (UTF-8 BOM CSV, 원본 KAMP 한글 컬럼)

| 파일 | 컬럼 | 규칙 |
|---|---|---|
| `history.csv` | 날짜, 시간, 15분, 30분, 45분, 60분, 평균, 생산량, 공장인원, 인건비, 기온, 풍속, 습도, 강수량 | 대상일 **전날 23:00** 에서 끝나는 연속 시간별, 하루 단위 완결. **최소 8일 · 권장 14일** (API 는 최대 62일) |
| `plan.csv` | 날짜, 시간, 생산량, 공장인원, 기온, 풍속, 습도, 강수량 | 대상일 0~23시 24행. 생산량 = ERP 계획, 기상 = 예보. **전력 컬럼이 있으면 거부**(누수) |

- 결측 허용: 풍속·강수량·공장인원 (정비 규칙: 강수량 → 0, 풍속·공장인원 → 시간 보간, 경고 출력)
- 계측정지(평균=0)는 이력 안에서만 보간한다(과거 값 기준, 인과적)
- θ(=187) 는 다시 계산하지 않는다 — 매니페스트 값이 피처 정의의 일부다

**출력** 24행

| 컬럼 | 의미 |
|---|---|
| `y_avg_pred`, `y_peak_pred` | 평균전력·peak15 예측 (kW) |
| `regime` | 게이트가 고른 가동 레짐 (0 기저 · 1 중간 · 2 가동) |
| `peak_label` | `y_peak_pred >= τ` — 제출 파일 `peak_pred_label` 과 같은 규칙 |
| `peak_prob` / `peak_prob_cal` | 피크 확률 (보정 전 / Isotonic 보정) |
| `peak_label_cls` | `peak_prob >= τ_cls` |
| `q10` · `q50` · `q90` | 분위회귀 (미보정 — 테스트 피복률이 목표에 못 미친다) |
| `pi_lo` · `pi_hi` | 최종모델 OOF 잔차 기반 90% 구간 (휴무일/운영일 반폭 따로) |

과거 구간 재생(검증용):

```bash
python -X utf8 -m serving replay --bundle outputs/models/full/eval --data data/okm_augumented_2021.csv \
       --start 2021-09-01 --end 2021-09-14 [--out replay.csv] [--window-days 14]
```

- `--bundle` 은 번들 **디렉터리 경로**다(bundle_id 가 아니다).
- `--out` 을 빼면 결과 CSV 를 표준출력으로 낸다. `--window-days` 를 빼면 원자료 첫날부터의 이력 전체를 쓴다(배치 파이프라인과 비트 동일).
- 평가 번들을 이력 전체로 돌리면 제출 파일과 같은 값이 나온다(`bash run.sh replay` = 이 명령 + `--out .cache/replay_eval.csv`).

## 5. REST API

```bash
KAMP_BUNDLE_DIR=/srv/kamp/bundles/<id> KAMP_API_KEY=<비밀> \
  uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port 8000
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /healthz` | `{status, bundle_id, role, train_last}` |
| `GET /v1/model` | 피처 계약·τ·휴일 커버리지·주의사항 |
| `POST /v1/predict/day-ahead` | `{target_date, history[8~62일], plan[24]}` → `{meta, rows[24]}` |

JSON 필드는 ASCII 별칭이다: `date hour p15 p30 p45 p60 avg prod headcount labor temp wind humid rain`
(계획은 전력 필드 없이 `date hour prod headcount temp wind humid rain`).

- 기동 시 번들을 즉시 검증·로드한다. 실패하면 **뜨지 않는다**(fail-fast).
- 워커 1개. 번들 교체는 재시작으로 한다(핫 리로드 없음).
- `KAMP_API_KEY` 가 있으면 `/v1/*` 에 `X-API-Key` 헤더가 필요하다.
- 본문 2MB 초과 → 413, `Content-Length` 없는 요청(chunked) → 411. 모든 수치는 유한값·물리 범위 검증, 모르는 필드는 거부.
- 모든 오류는 한 봉투 `{"error": {code, message, detail?, request_id}}` (404·500 포함). 응답 헤더 `X-Request-ID`.
- 요청마다 stderr 에 JSON 로그 1줄(요청 id·경로·상태·지연). 페이로드는 남기지 않는다.

## 6. 오류 코드

| 코드 | HTTP / CLI | 뜻과 대처 |
|---|---|---|
| `SCHEMA`, `HISTORY_SCHEMA`, `PLAN_SCHEMA` | 422 / 2 | 필드 누락·숫자 아님·NaN/inf·범위 밖·존재하지 않는 날짜·계획 컬럼 24시간 전부 결측 |
| `PLAN_HAS_POWER` | 422 / 2 | 계획에 대상일 전력이 있다(누수) — 빼고 보내라 |
| `PLAN_INVALID` | 422 / 2 | 계획이 대상일 0~23시 24행이 아니다 |
| `HISTORY_INVALID` | 422 / 2 | 빠진 시각·중복·전날 23시에 끝나지 않음·하루 미완결 |
| `HISTORY_TOO_SHORT` / `HISTORY_TOO_LONG` | 422 / 2 | 8일 미만 / API 62일 초과. `detail.required_start` 참고 |
| `LENGTH_REQUIRED` · `BODY_TOO_LARGE` · `UNAUTHORIZED` · `NOT_FOUND` | 411 · 413 · 401 · 404 | Content-Length 없음 · 본문 2MB 초과 · API 키 · 없는 경로 |
| `METHOD_NOT_ALLOWED` | 405 | 허용되지 않은 메서드 (예: `GET /v1/predict/day-ahead`) |
| `SHUTDOWN_RUN_TRUNCATED` | 422 / 2 | 대상일이 휴무인데 휴무 연속구간이 이력 시작에서 잘렸다 — 더 이른 이력을 보내라 |
| `HISTORY_EDGE_OUTAGE` | 422 / 2 | 이력 시작의 계측정지가 피처 참조 구간(최근 7일)까지 이어진다 |
| `CALENDAR_NOT_COVERED` | 422 / 2 | 대상일 연도의 공휴일이 `config/holidays_kr.json` 에 없다 |
| `FEATURE_INVALID` | 422 / 2 | 대상일 피처에 결측 — 입력을 확인하라 |
| `BUNDLE_NOT_FOUND` · `BUNDLE_SCHEMA` · `BUNDLE_INTEGRITY` · `BUNDLE_CONTRACT` | 기동 실패 / 3 | 번들 없음·형식 오류·파일 또는 **매니페스트** 손상·변조·피처/게이트 계약 불일치 |
| `HOLIDAYS_INVALID` | 기동 실패 / 3 | 운영 휴일표 JSON 형식 오류·잘못된 날짜·명시한 파일 없음 |
| `FAST_BUNDLE` | 기동 실패 / 3 | 테스트용 축소 번들 — `full/` 번들을 쓰라 |
| `VERSION_MISMATCH` | 기동 실패 / 3 | lightgbm·numpy·pandas 버전이 학습 때와 다르다 |
| `SELFTEST_FEATURES` · `SELFTEST_PREDICTION` | 기동 실패 / 3 | 이 환경에서 피처·예측이 학습 때와 다르다 |
| `PREDICTION_NONFINITE` · `INTERNAL` | 500 / 1 | 예측값이 유한하지 않다 · 처리되지 않은 오류(서버 로그의 request_id 로 추적) |

> 무결성 검사는 **손상·실수로 인한 변경**을 잡는다. 번들 디렉터리에 쓸 수 있는 사람은 매니페스트까지 다시 쓸 수 있으므로,
> 보안은 서비스 계정에 번들 디렉터리 **읽기 전용** 권한만 주는 것으로 지킨다.

## 7. 공휴일 — 운영자가 관리한다

학습에 쓴 `HOLIDAYS_2021` 은 2021-08-16 까지만 있다. `serving/config/holidays_kr.json` 이
연도 커버리지(`coverage_years`)와 날짜를 보충한다(현재 2021년 전체). 새 연도를 서비스하려면
**공식 출처로 확인한** 공휴일·대체공휴일·임시공휴일을 추가하고 연도를 `coverage_years` 에 넣는다.
커버리지 밖 대상일은 `CALENDAR_NOT_COVERED` 로 거부한다 — 휴일을 모르는 채 조용히 평일로
예측하는 것보다 안전하다.

다른 휴일표를 쓰려면 CLI `--holidays <json>` 또는 환경변수 `KAMP_HOLIDAYS_FILE`(API 포함)로 지정한다.
지정한 파일이 없거나 형식이 틀리면 `HOLIDAYS_INVALID` 로 기동하지 않는다.

## 8. 한계 (모델 자체)

- 학습 데이터가 2021-01~09 뿐이다. month 10~12 와 다른 해는 **외삽**이다 (`/v1/model` 의 caveats).
- 배포 번들의 τ·τ_cls·보정기·구간 반폭은 평가 번들의 교차검증 값을 상속한다(재학습 구간에는 CV fold 가 없다).
- 기상은 학습 때 실측을 예보의 대리로 썼다. 운영에서는 예보 오차만큼 성능이 떨어질 수 있다.
- 새 데이터로 재학습하려면 노트북 파이프라인을 새 데이터에 맞게 고쳐야 한다(2021 데이터에 묶인 게이트·상수가 있다).

## 9. 테스트

| 명령 | 범위 | 시간 | outputs |
|---|---|---|---|
| `bash run.sh serving-test` | 서빙 단위·코드생성 테스트 (파이프라인 없음) | 수 초 | 건드리지 않는다 |
| `bash run.sh serving-test-pipeline` | 번들·리플레이·API 테스트 (FAST 파이프라인 실행) | 수 분 | ⚠️ **FAST 결과로 덮어쓴다** — 끝나면 FULL 을 다시 돌린다 |

API 테스트(`tests/test_serving_api.py`)는 fastapi·httpx 가 없으면 통째로 skip 된다. `pip install -r serving/requirements.txt` 를 먼저 한다.
`serving/_core.py` 가 src 와 동기인지는 `python -X utf8 tools/build_serving_core.py --check` 로 따로 확인할 수 있다.
