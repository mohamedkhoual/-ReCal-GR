# ReCal-GR

Implementation code for **ReCal-GR: Recovery-Aware Generative Remediation with Assistance-Intensity Control**
(International Journal of Intelligent Engineering and Systems).

ReCal-GR estimates the probability that a learner recovers after a failure (Recovery@3), maps it to an
assistance level with the Recovery Calibration Index (RCI, a cost-sensitive Bayes decision rule), selects a
remediation action within that level and constrains an LLM to generate the corresponding remediation.

## Installation
```bash
pip install -r requirements.txt
# generation only (GPU): pip install transformers accelerate torch
```

## Data
Download the MOOC-Radar fine-grained release (`student-problem-fine.json`) and `problem.json` from
https://github.com/THU-KEG/MOOC-Radar. The dataset is not redistributed here.

## 1. Cohort, recovery model and RCI decision layer
```bash
python run_all.py --fine student-problem-fine.json --problems problem.json --out results
```
Expected key numbers: 165,837 incorrect interactions; 39,149 Recovery@3 episodes from 6,463 learners;
recovery rate 0.886; logistic regression out-of-fold ROC-AUC 0.814; tercile thresholds 0.900 / 0.958;
cost-derived thresholds 0.50 / 0.80; expected cost 0.181 (cost-derived) vs 0.423 (terciles).

```bash
python revision_analyses.py --interactions interactions.pkl --out results/revision
```
Thresholds derived on training folds, history-rule and shuffled comparators with the same region shares,
neighbouring thresholds, retained-vs-excluded comparison, elapsed time to recovery, confusion matrix.
Cross-session horizons: `cohort.build_cohort(df, k=3, min_gap_h=1)` (or 24).

```bash
python rci_plus_ablation.py results/cohort_scored.pkl
```
Ablation of the RCI (history modifier and uncertainty band).

## 2. Generative evaluation
```bash
python genkit/sample_cases.py results/cohort_scored.pkl problem.json genkit   # 60 cases, one per learner
python genkit/generate.py genkit/cases.jsonl genkit/outputs.jsonl             # Qwen2.5-3B-Instruct, greedy
python genkit/make_rating_sheets.py genkit/cases.jsonl genkit/outputs.jsonl genkit/sheets
python genkit/merge_ratings.py genkit/cases.jsonl genkit/sheets ratings_long.csv
```

## 3. Statistical analysis of the human ratings
```bash
python confirmatory_final.py        # confirmatory study: expects the rating files in data_confirmatory/
python pilot_supplement.py pilot_outputs.csv rater1.xlsx rater2.xlsx results/pilot   # pilot study
```

## Repository layout
- `recal/cohort.py` — failure episodes, eligible opportunities, Recovery@k labels, leakage-free history features
- `recal/predict.py` — learner-grouped cross-validation, baselines, calibration metrics, learner-level bootstrap
- `recal/rci.py` — RCI as a cost-sensitive Bayes rule, within-region action policy, offline policy evaluation
- `recal/prompts.py` — generation prompts (C1–C5, C4-M, C4-H, C4-X) and self-check prompt
- `recal/stats_human.py` — Friedman / Kendall's W, Wilcoxon with rank-biserial correlation, Holm, mixed model, McNemar, weighted kappa
- `genkit/` — case sampling, generation, rating sheets and rating merge
- `run_all.py`, `revision_analyses.py`, `rci_plus_ablation.py` — training and evaluation of the recovery model and RCI
- `confirmatory_final.py`, `pilot_supplement.py`, `pilot_analysis.py` — statistical analysis of the human ratings

## Hyperparameters
| Component | Setting |
|:--|:--|
| Features | 9 pre-failure features (6 numerical, 3 categorical); median imputation + standardization; one-hot encoding (`min_frequency=3`) |
| Cross-validation | `GroupKFold(n_splits=5)` grouped by learner; out-of-fold predictions |
| Logistic regression (selected) | L2, `C=1.0`, `max_iter=2000` |
| LightGBM | `n_estimators=100`, `learning_rate=0.05`, `num_leaves=15`, `max_depth=4`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1` |
| XGBoost | `n_estimators=100`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1` |
| Platt scaling (comparison) | `CalibratedClassifierCV(method="sigmoid", cv=3)` |
| Recovery target | Recovery@k with `k=3` (sensitivity: k = 1, 2, 5; minimum gap 1 h / 24 h) |
| RCI cost matrix C* | O = (0, 0.25, 1.25), U = (2, 1, 0), giving thresholds τ1 = 0.50, τ2 = 0.80 |
| Action policy | `kappa=2` consecutive failures, `theta=0.5` prior concept accuracy |
| Generation | Qwen2.5-3B-Instruct, greedy decoding (`do_sample=False`), `max_new_tokens=192`, `repetition_penalty=1.05` |
| Bootstrap | 2,000 learner-level replicates (prediction, policies); 5,000 case-level replicates (human ratings) |

## Reproducibility
Random seeds: `random_state=42` everywhere (GroupKFold is deterministic); bootstrap seed 42; shuffle seeds 7 / 42.
Tested with Python 3.13, pandas 2.x, scikit-learn 1.9, lightgbm 4.7, xgboost 3.4, statsmodels 0.15, scipy 1.x.
No pretrained weights are needed: the recovery estimator is a logistic regression trained by `run_all.py`,
and the generator is the public `Qwen/Qwen2.5-3B-Instruct`.
