# 포드 제출 패키지 (2026-09-25)

보고서 6장이 기술하는 제출 구성: **노트북 1개 + 데이터 + 실행 스크립트 + `serving/`**.
노트북은 `tools/build_experiment_notebook.py` 가 같은 `src/` 에서 조립한다(1부 환경과 정의 · 2부 실험).

## 만들기

```bash
python -X utf8 experiments/2026-09-25_pod_package/make_pod_package.py --out <만들 폴더>
# → <만들 폴더>/ 와 <만들 폴더>.zip (19개 파일, 약 0.28 MB), 개인정보 스캔 포함
```

| 넣는 것 | 출처 |
|---|---|
| `자원최적화_실험전용.ipynb` (361셀, 코드 272) | 생성기. 빠지는 절: 6.7·8.5·9장·10.2~10.4·10.6·최종 게이트 (보고서·제출물 생성). **6.6절과 10.5절(모델 저장)은 포함** |
| `data/okm_augumented_2021.csv` | 저장소 `data/` |
| `README.md` · `run.sh` · `check_env.sh` · `setup_pod.sh` · `requirements.txt` | 이 폴더 `pod_files/` (포드 전용 판) |
| `serving/` (12개 파일) | 저장소 `serving/` (`CLAUDE.local.md`·`__pycache__` 제외) |

## 2026-09-25 기록

- 개발 PC 에서 FULL 로 미리 돌렸다(44분).
  - 코드셀 272개 실행, 오류 0.
  - 표 106개·그림 39장.
  - 예측 파일이 정본과 **바이트 동일**하다.
  - 번들 ID `eval-full-7be5289680f8`·`deploy-full-d8293292a723` 가 정본과 같다.
  - `serving verify`·`replay`(MAE 5.2199)가 통과했다.
- 포드와 같은 라이브러리 버전(pandas 2.1.4·numpy 1.26.4·lightgbm 4.5.0)의 가상환경에서 서빙 단위 테스트 76개가 통과했다.
- 이 스크립트로 다시 만든 패키지를 당시 만든 `pod_실험전용_v2` 와 비교했다. 파일 19개 중 18개가 바이트 동일하고, 노트북은 **셀 내용이 모두 같으며 무작위 셀 ID 만 다르다**(생성기가 빌드마다 새 ID 를 붙인다).
- 패키지 폴더 안에서 셸을 돌리면 도구 상태 폴더(`.omc/`)가 생기는데, 여기에 계정명이 들어간 적이 있다. zip 은 점 폴더를 빼고 만든다.
- **남은 일: 포드에서 FULL 재실행**(`KAMP_FAST=1 bash run.sh` → `nohup bash run.sh > full_run.out 2>&1 &`, 45~75분).
  - 포드는 lightgbm 4.5.0 이라 번들 ID 는 달라지는 게 정상이다.
  - `outputs/predictions_test_336h.csv` 와 `outputs/tables/ch2_lag_decomposition.csv` 가 위 결과와 같은지 확인한다.
