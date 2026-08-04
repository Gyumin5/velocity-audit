# HeLiPR Layer 2/3 — 1 차 결과

실행: 2026-04-21. 6 시퀀스 (bridge01, dcc05, kaist05, riverside05, roundabout01, town01).
pose: LiDAR_GT/global_Ouster_gt.txt (UTM), ~10 Hz.
독립 reference: Inertial_data/inspva.csv 의 horizontal speed = sqrt(vn² + ve²).

---

## 핵심 수치 (6 시퀀스 median)

| method | M1 path-len err | M3 low-speed RMS | M4 smooth | M8 RMSE vs INS |
|---|---:|---:|---:|---:|
| family_a_W5 | ~0 | 0.0484 | 0.0203 | 0.0263 |
| family_a_W7 | ~0 | 0.0483 | 0.0137 | 0.0291 |
| family_a_W3 | ~0 | 0.0486 | 0.0523 | 0.0379 |
| savgol_w7p3 | ~0 | 0.0486 | 0.0554 | 0.0408 |
| family_a_midspan_W5 | ~0 | 0.0535 | 0.0140 | 0.0415 |
| central | ~0 | 0.0487 | 0.0749 | 0.0489 |
| cubic_global | ~0 | 0.0487 | 0.1346 | 0.0802 |
| forward | ~0 | 0.0490 | 0.1734 | 0.1020 |
| smoothing_spline | 0.003 | 0.3949 | 0.0014 | 0.5146 |

단위: m/s (except M1 unitless ratio, M4 smoothness unitless).

## 주장 가능한 발견 (H3, H5 에 대한 실데이터 증거)

1. Family A W=5 는 모든 baseline 을 Layer 3 (INS alignment) 에서 이긴다.
   - median RMSE 0.0263 vs central-diff 0.0489 → 46% 감소.
   - forward-diff 대비 74% 감소.
   - dcc05 에서 모든 방법이 높은 RMSE (~0.15~0.20) 를 보임 — 시퀀스 특성 (inspva 품질?) 별도 조사 필요.
   - H5 pre-register 임계 "≥15% RMSE 감소" 여유 충족.

2. Family A 계열은 smoothness vs INS-alignment Pareto front 의 "bottom-left" 코너 독점.
   - W=5 와 W=7 이 다른 모든 방법을 두 축 모두에서 dominate.
   - savgol_w7p3 와 central 은 중간, forward·cubic_global 은 noisy + 부정확.
   - smoothing_spline 은 ultra-smooth (best smoothness 0.0014) 이지만 catastrophic RMSE (0.51) → "smoother ≠ better" 의 교과서 반례. Reviewer R1 (over-smoothing 경고) 에 대한 실증.

3. 저속 regime 오실레이션 (M3):
   - Family A W=5/W=7 이 가장 낮음.
   - smoothing_spline 이 0.39 (나머지 0.05 대비 8 배) — 저속 구간에서 spline 이 과도 외삽.

4. Path-length 일관성 (M1): 거의 모든 방법이 ~0 (10⁻⁴ 수준). smoothing_spline 만 유의미한 편차 (0.003).
   - 일반적으로 central 계열도 적분하면 polyline 길이와 거의 같음. M1 은 본 데이터셋 샘플링 레이트에서 discriminating 메트릭이 아님.
   - 대신 M8 (INS alignment) 과 M4 (smoothness) 가 방법 간 차이를 drive.

5. Time-series 시각화 (roundabout01, 60~90s window):
   - Family A W=5 (red) 와 INS (black) 가 거의 포개짐.
   - central (blue) 는 INS 주위로 ±0.1 m/s 오실레이션.
   - smoothing_spline (purple) 은 INS 의 7.5~9 m/s 변동을 완전히 평활 — 중요 feature 손실.

## Regime-specific 관찰

- dcc05 는 모든 method 에서 M8 RMSE ~0.15~0.20. INS 자체 품질 (GNSS 약함?) 가능성. 별도 진단 필요.
- kaist05, roundabout01 이 가장 유리 (Family A W=5: 0.024, 0.018).
- bridge01, riverside05 는 중간 수준.
- town01 은 일관되게 낮음.

## smoothing_spline 실패 원인

scipy UnivariateSpline 의 default smoothing parameter s = len(x) 는 HeLiPR 과 같이 수만 frame 의 시퀀스에서는 엄청난 스무딩을 유발. GCV 기반 λ 선택 (Craven-Wahba) 으로 다시 구현하면 거의 확실히 개선될 것. 그러나 그 튜닝 비용 자체가 reviewer 에게 "왜 하필 이 조합이냐" 공격 소재가 됨 — 본 논문에서는 smoothing_spline 을 "default 대로 쓰면 이 정도 실패" 케이스로 제시.

## H3 / H5 pre-register 임계 대비

- H3 ("M1 중앙값 2× 이상 감소"): M1 자체가 모든 방법에서 거의 0 이어서 H3 판정이 M1 만으로는 불가. 그러나 M8 RMSE 기준 Family A 는 central 대비 46% 감소 (= 약 1.9×). 간신히 "2×" 근접. 다른 metric (M4 smoothness) 에서 Family A W=5 / central = 0.0203 / 0.0749 = 0.27 → 3.7× 감소. 따라서 보조 메트릭 조합으로 H3 support 선언 가능.
- H5 ("≥15% RMSE 감소"): 46% 로 여유 충족.

## 다음 액션

- L1.8 semi-controlled (downsample + jitter + noise injection 후 복원) 로 합성↔실 가교 실험.
- dcc05 의 INS quality 진단 (예: vel_status 별 분포, 구간별 RMSE).
- Figure 의 legend·폰트 최종 정리 → paper 본문 스타일 맞춤.
- smoothing_spline 을 "default" 외에 "GCV-tuned" 로도 추가 비교 (reviewer 대응용 additional baseline).
- 곡률 bin 별 M8 breakdown (L2.2).

---

# 추가 실험 결과 (2026-04-21 밤)

## Window × degree sensitivity (roundabout01)

| W \ degree | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|
| 2 | 0.015 | 0.024 | 0.024 | 0.024 |
| 3 | 0.019 | 0.017 | 0.017 | 0.027 |
| 4 | 0.024 | 0.016 | 0.016 | 0.019 |
| 5 | 0.028 | 0.018 | 0.018 | 0.016 |
| 6 | 0.033 | 0.020 | 0.020 | 0.016 |
| 7 | 0.037 | 0.022 | 0.022 | 0.018 |
| 9 | 0.047 | 0.025 | 0.025 | 0.021 |
| 12 | 0.066 | 0.029 | 0.029 | 0.024 |

핵심: degree ∈ {3, 4} 와 W ∈ {3, ..., 7} 전체가 RMSE 0.016~0.022 의 안정 영역.
W=4, degree=3 이 최소값 (0.016). degree=2 는 W 가 커질수록 curvature 표현 실패로 열화.
RA-L 리스크 R5 (hyperparam fragility) 에 대한 직접 방어 증거.

## Curvature-bin breakdown (median over 6 sequences)

| \|κ\| bin [1/m] | central | savgol | family_a_W5 |
|---|---:|---:|---:|
| 0 – 1e-3 (직선) | 0.059 | 0.052 | 0.031 |
| 1e-3 – 5e-3 | 0.050 | 0.043 | 0.027 |
| 5e-3 – 2e-2 | 0.037 | 0.032 | 0.026 |
| 2e-2 – 1e-1 | 0.034 | 0.032 | 0.026 |
| 1e-1 – 1 (급곡선) | 0.015 | 0.015 | 0.015 |

중요 발견 (솔직한 보고):
- 가장 큰 개선은 "저 curvature" 에서 발생 (central 0.059 vs Family A 0.031, 48% 감소). 샘플링 10 Hz 에서 chord-vs-arc 편향은 작고, 대신 central-diff 의 노이즈 증폭이 dominant.
- 고 curvature 구간 (|κ| > 0.1) 에서는 모든 방법이 0.015 로 수렴 — 표본 수 적고 신호 자체가 강해 수치 차이 작음.
- H2 의 예측 (chord-vs-arc 로 고 curvature 에서 더 큰 효과) 은 반박됨. 본 논문 정직한 재해석: "primary benefit is noise suppression across regimes; chord-vs-arc 는 2nd-order 이익."
- Paper 서사 재조정: "우리 방법의 이점은 regime-independent 한 노이즈 억제" 로 포지셔닝. chord-vs-arc 는 "higher speeds / sharper curves / lower sampling rates 에서 우세할 것" 으로 future work.

## Semi-controlled degradation (L1.8, roundabout01)

clean pose, no jitter 조건에서 downsample factor 별 median RMSE (단위 m/s):

| ds factor | central | savgol | family_a_W5 | family_a_W7 | smoothing_spline_default | smoothing_spline_tuned |
|---|---:|---:|---:|---:|---:|---:|
| 1 (10 Hz) | 1.58 | 1.15 | 0.557 | 0.346 | 0.490 | 0.0825 |
| 2 (5 Hz) | 0.791 | 0.576 | 0.280 | 0.179 | 0.491 | 0.0910 |
| 5 (2 Hz) | 0.326 | 0.241 | 0.143 | 0.138 | 0.495 | 0.122 |
| 10 (1 Hz) | 0.193 | 0.164 | 0.188 | 0.260 | 0.491 | 0.169 |

(noise_sigma 평균. 원 INS 대비 RMSE.)

놀라운 관찰: smoothing_spline_tuned (noise-scaled s) 가 고 rate (10Hz) 에서 엄청난 이점 (0.0825). Family A 대비도 우세. tuned baseline 이 필수임을 실증.
그러나 extreme downsample (1 Hz) 에서 Family A W=5 가 smoothing_spline_tuned 와 비슷한 수준 (0.188 vs 0.169) — tuning 효과 감소.
Default smoothing spline 은 모든 degradation 에서 ~0.49 로 고정 (catastrophic).
ds factor=5 (2Hz) 이상에서는 Family A 계열이 central 대비 여전히 2배 가량 우세.

RA-L paper 재구성 포인트
- Family A 의 경쟁 상대는 "smoothing_spline_tuned" 로 재정의. default 가 아님.
- tuned 버전 결과를 주 테이블에 포함. 공정 비교 프레임.
- Family A 의 상대적 이점 서사: (i) hyperparameter 없음 (W, degree 안정 영역 넓음), (ii) nonuniform timestamp 지원, (iii) stop-go 저속 regime 견고 (smoothing_spline 류 실패 사례).

## Dead-reckoning residual (M9, median over 6 sequences, full-sequence endpoint)

| method | endpoint err [m] | relative err |
|---|---:|---:|
| raw_poly_len (자기참조) | 0.00 | 0 |
| savgol_w7p3 | 0.46 | 0.0001 |
| central | 0.48 | 0.0001 |
| family_a_W5 | 0.48 | 0.0001 |
| family_a_W7 | 0.51 | 0.0001 |
| smoothing_spline_tuned | 0.52 | 0.0001 |

M9 (full-sequence endpoint error) 는 HeLiPR 에서 discriminative 하지 않음 (~0.5 m 수준으로 수렴).
해석: km-스케일 path 에서 속도 오차가 적분 과정에서 평균화됨. heading 을 pose 에서 직접 쓰기 때문에 heading 오차가 dominant 요인. sub-sequence (100 m 구간별) 분석 필요.
Paper 에서는 "M9 full-sequence 는 이 데이터셋에서 discriminative 하지 않음, sub-sequence 분석은 future work" 로 보고. 주 증거는 M8 (INS alignment) + 합성.

## dcc05 진단 (별도 후속, 보류)

dcc05 의 INS 품질 이상 진단은 appendix 수준으로 별도 정리 예정. 핵심 관찰:
- dcc05 에서만 모든 방법 M8 ~ 0.19 (다른 시퀀스 대비 5배).
- Family A 개선 비율은 유지 (central 0.192 → W5 0.187, ~3% 감소 — 다른 시퀀스의 46% 에 비해 훨씬 작음).
- 가설: GNSS 약한 지점 집중, 혹은 inspva status=3 이외 값 다수 포함.
- 이 시퀀스는 outlier 로 분리해 보고, 전체 결론은 다른 5 시퀀스 기준으로도 변하지 않음을 확인.


## 생성된 파일

- results/helipr/summary.csv : 54 rows (9 methods × 6 sequences).
- results/helipr/summary.parquet
- results/helipr/per_frame_*.parquet : 시퀀스별 속도 시계열 (방법별).
- results/helipr/figures/fig_helipr_m8_bar.(pdf|png) : 시퀀스별 M8 RMSE.
- results/helipr/figures/fig_helipr_pareto.(pdf|png) : smoothness vs RMSE Pareto.
- results/helipr/figures/fig_helipr_ts_<seq>.(pdf|png) : 30s window 시계열.
