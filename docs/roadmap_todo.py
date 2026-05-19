"""
Engineering roadmap — model comparison & regression.

Documentation only (not imported at runtime). Check items as [x] when merged.
Existing: compare_periods.py (history length ablation), XGBoost production path.

Priority: ship a correct evaluation harness before adding models.
Defer: LSTM, full hyperparameter grids, multi-ticker automation.
"""

# =============================================================================
# MILESTONE 0 — Decisions (½ day, do before coding)
# =============================================================================
#
# [ ] 0.1 Write 1-page design note (README section or docs/MODELING.md)
#       Decide and record:
#         • Primary task for report: classification | regression | both
#         • Regression target: next-day log-return (recommended) vs raw Adj Close
#         • Production model after compare: keep XGBoost unless table shows clear win
#         • Success metrics: test accuracy/F1 + MAE/RMSE + direction-from-price;
#           backtest is illustrative only (not optimization target)
#       Done when: another dev can read it and know what to implement.
#
# [ ] 0.2 Agree evaluation contract (no code yet)
#       • Fixed split: 70 / 15 / 15 chronological (match preprocess.py)
#       • Scaler: fit on train only, transform val/test
#       • Test set touched once per experiment (final row in report table)
#       • Val set: early stopping + hyperparameter selection only
#       Done when: listed in design note; referenced by compare script.
#
# =============================================================================
# MILESTONE 1 — Shared training/eval module (1 day)
# =============================================================================
# Why first: compare_periods.py and train_bundle.py duplicate split/train logic.
# Extract once; all future scripts call the same functions → fair comparisons.
#
# [ ] 1.1 Add src/experiment.py (or src/training_pipeline.py)
#       Functions:
#         build_dataset(ticker, benchmark, period) -> feat, X, y, feature_cols, raw
#         split_and_scale(X, y) -> Xtr, Xva, Xte, y_*, scaler
#         eval_classification(model, scaler, test_df, feature_cols) -> metrics dict
#       Refactor train_bundle.py to use it (behavior unchanged).
#       Done when: existing `python scripts/run.py train MSFT` still works; diff is refactor only.
#
# [ ] 1.2 Add eval_test_window() helper
#       Mirror compare_periods: append_next_session_prices, test slice indices,
#       classification_metrics_summary + optional backtest dict.
#       Done when: compare_periods imports helper; output numerically unchanged on MSFT 10y.
#
# [ ] 1.3 Smoke test script: scripts/smoke_train.py (or pytest)
#       Assert: feat rows > 100, no NaN in X_train, label in {0,1}.
#       Done when: runs in <60s on 2y period without Finnhub (or mocked).
#
# =============================================================================
# MILESTONE 2 — Model comparison, classification (1–2 days)
# =============================================================================
# Scope v1: three baselines + current champion. Add LightGBM only if v1 stable.
#
# [ ] 2.1 src/model_registry.py
#       Registry: name -> factory(train) + requires_early_stopping flag.
#       v1 models:
#         • logistic     — LogisticRegression(max_iter=2000, class_weight balanced)
#         • rf           — RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1)
#         • sklearn_gbc  — GradientBoostingClassifier(n_estimators=200, max_depth=4)
#         • xgboost      — existing init_xgb_classifier + train_xgb_baseline
#       Done when: each returns a fitted sklearn-compatible classifier with .predict / .predict_proba.
#
# [ ] 2.2 scripts/compare_models.py
#       Args: ticker, --period, --models logistic,rf,xgboost, --out results.json
#       Loop: same dataset from 1.1, train on train, early stop on val where applicable,
#             evaluate on test once, append row to results list.
#       Print: sorted table (accuracy, f1, val_logloss if available).
#       Done when: `python scripts/compare_models.py MSFT --period 5y` prints ≥3 rows in <5 min.
#
# [ ] 2.3 Wire scripts/run.py compare-models → compare_models.main
#       Done when: documented in README one-liner.
#
# [ ] 2.4 Add lightgbm (optional, size S)
#       requirements.txt + registry entry lgbm.
#       Skip if install fails on target machine; note in report.
#       Done when: fourth row appears in compare table.
#
# [ ] 2.5 Export for report
#       --out-csv models/compare_MSFT.csv with columns: model, period, accuracy, f1, ...
#       Done when: CSV pasteable into coursework appendix.
#
# Out of scope for v1:
#   • Training all models inside API bundle
#   • Statistical significance tests (mention as future work in report)
#
# =============================================================================
# MILESTONE 3 — XGBoost tuning (½–1 day, after Milestone 2)
# =============================================================================
# Separate script/experiment — do not silently change production defaults until reviewed.
#
# [ ] 3.1 scripts/tune_xgb.py
#       RandomizedSearchCV + TimeSeriesSplit(n_splits=3) on TRAIN only (or train+val
#       with inner split — never test). Small grid (~20–40 trials), metric: log-loss.
#       Params: max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda.
#       Done when: prints best_params_ and best val score; saves tune_report.json.
#
# [ ] 3.2 Single test evaluation
#       Load best params → train via experiment.py → eval_test_window once.
#       Compare test metrics to default XGB from 2.2 in report table.
#       Done when: README explains "tuned vs default" row.
#
# [ ] 3.3 (Optional) Promote tuned params to init_xgb_classifier defaults
#       Only if test F1 improves AND val curve looks stable.
#       Done when: commit message cites compare + tune artifacts.
#
# =============================================================================
# MILESTONE 4 — Regression vertical slice (2–3 days)
# =============================================================================
# Ship end-to-end with ONE regressor before comparing six. Recommended target:
#   y = log(target_Adj_Close.shift(-1) / target_Adj_Close)  # next-day log return
# UI derives: next_price = today_adj * exp(pred_logret), pct = (exp(pred)-1)*100
#
# [ ] 4.1 features.py — labels
#       • Target_Next_LogReturn (train/regress)
#       • Target_Next_Adj_Close (evaluate MAE in dollars if needed)
#       • Target_Direction (derived, keep for direction accuracy)
#       • keep_incomplete_target: last row NaN on targets, ok for predict
#       Done when: export CSV shows new columns; row count unchanged vs classification.
#
# [ ] 4.2 preprocess.py
#       • split_features_and_target(..., target_column=...)  # already has param
#       • Do NOT put target in StandardScaler with X (leakage risk / wrong semantics)
#       Done when: regression y is raw log-return; X scaling unchanged.
#
# [ ] 4.3 evaluate.py — regression_to_user_outputs()
#       Inputs: pred_log_return, today_adj_close
#       Outputs: next_price, pct_change, direction (0/1)
#       Metrics: regression_metrics_summary(y_true, y_pred) -> mae, rmse, mape, r2
#       Done when: unit test or doctest on synthetic numbers.
#
# [ ] 4.4 Train ONE production regressor: XGBRegressor
#       src/models/regression_model.py — mirror baseline_model early stopping pattern
#       train_bundle.py: --task regression (default classification for backward compat)
#       Bundle keys: task, target_column, model, scaler, feature_columns, ...
#       Done when: bundle loads; predict produces float vector length 1.
#
# [ ] 4.5 app_api.py / predict_cli / app_ui
#       Response schema v2 (backward compatible):
#         direction, direction_probability (classification only),
#         predicted_next_close, predicted_pct_change, task, model_type
#       Branch on bundle["task"].
#       Done when: UI shows three numbers; classification bundles still work.
#
# [ ] 4.6 compare_models.py --task regression
#       v1 regressors: ridge/linear, rf, xgboost_reg (skip LGBM until classification stable)
#       Table: model, mae, rmse, direction_accuracy_derived
#       Done when: one command produces regression table for report.
#
# =============================================================================
# MILESTONE 5 — Expand regression model zoo (1 day, optional)
# =============================================================================
#
# [ ] 5.1 Add to registry: LinearRegression, RandomForestRegressor, GBR, LGBMRegressor
# [ ] 5.2 Report: short paragraph which model wins on which metric; no overclaiming
#
# =============================================================================
# MILESTONE 6 — LSTM spike (1–2 days, GO / NO-GO)
# =============================================================================
# Treat as research spike, not production requirement.
#
# [ ] 6.1 Spike criteria (define before coding)
#       GO if: test direction accuracy > best tabular by ≥1 pp AND training <30 min CPU.
#       NO-GO: document "tabular sufficient" and stop.
#
# [ ] 6.2 Minimal LSTM in src/models/sequence_model.py
#       • Window: 20 sessions × n_features
#       • Train/val split aligned to dates (no shuffle)
#       • Compare on same test window as 2.2
#       Done when: GO/NO-GO decision recorded in report.
#
# Out of scope unless GO:
#   • LSTM in FastAPI bundle, FinBERT+sequence joint training
#
# =============================================================================
# MILESTONE 7 — Docs & coursework artifacts (½ day, continuous)
# =============================================================================
#
# [ ] 7.1 README: compare-models, tune_xgb, --task regression
# [ ] 7.2 FEATURES_REPORT: new target columns + formula for UI fields
# [ ] 7.3 Report table: include period=10y, ticker=MSFT, date of run, git commit hash
# [ ] 7.4 Honest limitations paragraph (noise, leakage controls, not financial advice)
#
# =============================================================================
# Suggested schedule (realistic for one developer)
# =============================================================================
#
#   Week 1: M0 → M1 → M2 (compare table in report)
#   Week 2: M3 → M4.1–4.5 (regression demo in API)
#   Week 3: M4.6, M5 if needed, M6 spike OR skip, M7 polish
#
# =============================================================================
# Definition of Done (project-level)
# =============================================================================
#
#   [ ] compare-models produces reproducible JSON/CSV for ≥4 classifiers on MSFT
#   [ ] tune_xgb documents whether tuning beat default on TEST (single evaluation)
#   [ ] Regression bundle predicts next_price + pct_change + direction on /predict
#   [ ] No secret keys in repo; .env unchanged
#   [ ] README lets a grader run train → compare → predict without reading source
#
# =============================================================================
# Explicitly NOT doing (avoid scope creep)
# =============================================================================
#
#   • Auto-trading or portfolio optimization
#   • Live Finnhub/Yahoo streaming
#   • Bayesian optimization / Optuna (unless M3 finishes early)
#   • Replacing static peer_maps with LLM ticker lists
#   • Multi-ticker batch training in CI
