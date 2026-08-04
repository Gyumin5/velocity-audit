# L1.0 Synthetic — 1차 발견

실행: 2026-04-21, seed 5, 5760 rows.

sweep: scenario ∈ {circle_r15_v5, circle_r5_v2, stop_go, straight_var}
× fs ∈ {2, 5, 10, 50} Hz × σ_p ∈ {0, 0.01, 0.05, 0.20} m × jitter ∈ {0, 5 ms} × 5 seed.

## 핵심 수치 (fs=10, σ_p=0.05, no jitter — median RMSE [m/s], 시나리오별)

| method | circle_r15 | circle_r5 | stop_go | straight |
|---|---:|---:|---:|---:|
| forward diff | 0.709 | 0.701 | 0.779 | 0.700 |
| central diff | 0.344 | 0.350 | 0.382 | 0.345 |
| cubic_global | 0.577 | 0.589 | 0.637 | 0.566 |
| savgol_w7p3 | 0.243 | 0.253 | 0.263 | 0.248 |
| smoothing_spline | 0.219 | 0.170 | 0.970 | 0.002 |
| family_a_W3 | 0.243 | 0.253 | 0.263 | 0.248 |
| family_a_W5 | 0.117 | 0.122 | 0.131 | 0.119 |
| family_a_W7 | 0.072 | 0.073 | 0.082 | 0.072 |
| family_a_midspan_W5 | 0.066 | 0.065 | 0.153 | 0.064 |

## 1차 발견 (H1, H3 pre-register 예상치에 대한 첫 근거)

1. Family A 의 의미는 "interpolating spline 이 아니라 local LS polynomial" 이어야 함.
   초기 구현 (scipy CubicSpline 보간) 은 노이즈 σ_p=0.05 m 에서 central-diff 보다 나빴다 (0.53 vs 0.34). 국소 다항 LS 로 바꾸자 정반대가 됨.
   이는 METHOD_CANDIDATES 의 문구 ("local spline") 를 "local polynomial LS fit" 으로 해석해야 한다는 근거. 본문 수식에 명시 필요.

2. Family A 는 모든 시나리오에서 중심차분·전방차분을 크게 이긴다.
   fs=10, σ_p=0.05 m 기준 median RMSE 비:
   - family_a_W7 / central ≈ 0.21 (약 5 배 개선)
   - family_a_W5 / central ≈ 0.35 (약 3 배)
   - H3 predict ("2× 이상 감소") 를 synthetic 기준에서는 여유 충족.

3. Smoothing spline 의 양면성.
   - straight 시나리오에서 RMSE 0.002 (극히 낮음) — over-smoothing 이 오히려 유리.
   - stop_go 시나리오에서 RMSE 0.97 (최악) — 저속 구간 edge 를 과도 평활.
   - 결론: smoothing spline 은 "smooth assumption 맞을 때만" 강함. 범용 대안이 아님. 본문 Related Work 에서 "왜 smoothing spline 단독 사용이 답이 아닌가" 주장의 경험적 근거.

4. midspan (A2) vs pointwise (A1).
   - smooth 시나리오에서는 midspan 이 pointwise W=7 보다 소폭 우월.
   - stop_go 에서는 midspan 이 pointwise W=7 보다 소폭 열세 (0.153 vs 0.082).
   - 결론: 기본 제안 메서드는 A1 (pointwise) + degree=3 + W=5~7. A2 는 보조로 보고.

5. SG (window 7, polyorder 3) 는 Family A W=3 과 사실상 동등.
   - SG 는 균일 샘플링 가정. Family A 는 비균일 허용.
   - 실데이터 nonuniform timestamp 에서 Family A 의 이점이 추가로 드러날 것.

6. forward-diff 는 모든 설정에서 최악. baseline 으로만 의미.

## 파생 결과물

- results.parquet / results.csv : 전체 5760 rows.
- summary.csv : scenario × method × fs × σ_p median + IQR.
- figures/fA_rmse_vs_fs.(pdf|png) : sampling-rate sweep.
- figures/fB_rmse_vs_noise.(pdf|png) : position-noise sweep.

## 다음 액션 (Phase 3 이후)

- 실데이터에서 midspan vs pointwise 재비교 (stop_go 케이스에서 midspan 열세 현상 재현 여부).
- smoothing spline 의 λ 를 GCV 로 제대로 선택하면 stop_go 케이스 성능이 개선되는지 확인.
- Family A 의 W 선택 heuristic 제안 (fs · Δt ≈ 0.5~1 s 이내).
- synthetic 에서 jitter 영향 sweep 결과를 별도 figure 로.
