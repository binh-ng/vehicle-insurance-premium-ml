# What changed since DataCon 2025

The original notebooks, report, slides and poster are kept unchanged in [`original_work/`](original_work) and [`docs/`](docs) (notebook outputs cleared). Everything below describes the expanded version in this repo and why each change was made.

### Data
1. **Both raw files are used.** The original project used only the 2014–2018 file (508,499 rows). The expanded version combines both Mendeley files (2011–2014 and 2014–2018), giving 801,609 policies on 288,676 vehicles after cleaning.
2. **Zero insured values are treated as liability-only cover, not missing data.** About 43% of policies have an insured value of 0. Their premiums sit on a few fixed tariff amounts, and the dataset author confirms liability cover is priced on a national tariff by vehicle type (Terefe, 2023b). The original MLR imputed these zeros with a group median. The expanded version keeps the zero and adds a `coverage` feature.
3. **Same-period claims are no longer predictors.** `claim_paid` / `insurance_claim` describe what happened *during* the policy, so they are not known when the premium is set. They are replaced by `prior_claim` (a claim on the vehicle's previous policy) and `prior_policies` / `customer_tenure`, which reflect the insurer's no-claim discount using only past information.
4. **New features:** vehicle age (instead of production year), policy length in days (short policies are kept and modelled rather than dropped), start season, and make (top 30 brands plus "Other").
5. **Currency label.** The report labelled amounts as USD. The CSV does not name a currency. The dataset author's paper says claim amounts were "originally the amount in the Ethiopian currency Birr" and were converted to USD for that paper's analysis (Terefe, 2023b), and the magnitudes in the file (for example, an insured value of 519,755 for a 2007 pickup) only make sense in Birr. The app labels amounts ETB and notes the uncertainty.
6. **Carrying capacity and seats are used with care.** The dataset author dropped them because they "are not correctly coded" (Terefe, 2023b). We keep them, clipped to plausible ranges, and the ablation in the app shows how much they add.

### Evaluation
7. **Split by vehicle.** The same vehicle appears in up to 17 policies (2.8 on average). A random row split puts about 82% of test rows on vehicles already seen in training. All expanded models use a split grouped by vehicle, and all tuning uses grouped cross-validation. (In practice the score difference turned out to be small, but the grouped split is the correct design.)
8. **One test set for every model.** The original Random Forest used a 10% sample and the MLR used a separately filtered dataset, so the two were not compared on the same policies. Now all 13 models share the same data, features and test set.
9. **Temporal holdout added:** train on policies starting before July 2017, test on later ones.
10. **Back-transform bias fixed.** Models fitted on log premium are converted back with Duan's (1983) smearing correction instead of a plain `exp()`, which predicts the median and understates the mean.
11. **More informative metrics.** Premiums range from under 100 to several million, so RMSE and R² on the raw scale are dominated by a few large fleet policies. Median % error, share of predictions within 20%, and R² on the log scale are added.

### Models
12. Expanded from 2 model families to 13 models: a tariff lookup baseline, OLS, Ridge, Lasso, Elastic Net, Gamma GLM, Decision Tree, Random Forest, HistGradientBoosting, XGBoost, LightGBM, a neural network (MLP), and a LightGBM + XGBoost ensemble. HistGradientBoosting, XGBoost and LightGBM are tuned with Optuna (Akiba et al., 2019); the decision tree and random forest with randomized search; the linear models with cross-validated alpha. All tuning uses folds grouped by vehicle. Tuned values are in `results/best_params.json`.
13. SHAP explanations replace impurity-based importance (impurity importance favours continuous, high-cardinality features).
14. A premium estimator with an 80% range (LightGBM quantile models) powers the Streamlit demo.

### Corrections to the original analysis
15. **Report comparison table:** the "MLR (Ridge)" row showed the polynomial model's scores (46.15% / 28.66%). The Ridge scores were 46.16% / 42.24%, which is what the text and poster say.
16. **Random Forest feature importance in the report came from the untuned model.** The report's top-5 list (insured value 0.6582, engine capacity 0.1115, ...) matches Alvin's working notebook, where the "tuned" section read `rf_model.feature_importances_` (the baseline forest) instead of `best_rf`. The final group notebook used `best_rf` correctly and printed different values (insured value 0.5076, engine capacity 0.1530, seats 0.0613, ...), but the report kept the earlier numbers.
17. **"Vehicle Usage 4.3%"** was the importance of one dummy column (`usage_Fare Paying Passengers`), not usage as a whole. Summing all usage dummies in the re-run tuned forest gives about 9%, which makes usage the third most important factor after insured value and engine capacity (`results/replication.json`).
18. **Interaction model:** `PolynomialFeatures` was applied to a matrix that already contained the `const` column, so every feature was duplicated as `const × feature`. That explains the condition number of 1e16 and the test R² of −56%, rather than interactions being inherently unstable.
19. **"Adjusted VIF after Ridge"** used a formula on the coefficients that is not a variance inflation factor. It is not used in the expanded version.
20. **Ridge gave the same result as OLS.** That is expected rather than a failure: with hundreds of thousands of rows and a few dozen features, the penalty barely changes the coefficients. The expanded version searches a wider grid (alpha from 0.0001 to 1,000) and still finds Ridge equal to OLS, so the report's claim that Ridge "improved generalization" over the additive model is dropped. Regularization only matters here if many interaction terms are added.
21. **Consumer advice:** "choose a lower insured value" now carries a caveat. Underinsuring usually reduces what you are paid after a loss.
22. **The low MLR R² was mostly a scale effect.** The MLR was fitted on log premium and scored on the premium scale after a plain `exp()`. On the log scale the model had an in-sample R² of 0.74 after removing influential points (statsmodels output in the final notebook), so it explained most of the variation in log premium; the 0.42 figure mostly reflects how it was scored. With smearing and the full feature set, Ridge reaches R² = 0.88 on log premium in the expanded version.
23. **Imputing zero insured values hurt the linear model.** On the same vehicle-grouped test set, Ridge with the original median imputation has a median error of 51.9%; keeping the zeros with a coverage flag cuts it to 28.6% (`results/imputation_check.csv`).
24. **Original Random Forest re-run.** `scripts/replicate_original.py` reruns the tuned forest with the original code path and best parameters. Over five different 10% samples, test R² ranges from 0.71 to 0.79, so the reported 0.742 is reproducible, but one 10% sample is not enough to compare models precisely.
25. **Original de-duplication removed half the data.** `drop_duplicates()` was applied after keeping only 9 columns, so renewals of the same vehicle with identical values were treated as duplicates. 244,848 of 508,499 rows remained before sampling. The expanded version drops only rows that are identical across all 16 raw columns.

### References
26. Colella & Jones (2023) was listed with the DOI of the XGBoost paper; replaced with the CAS link.
27. "Y. Yang (2025)" is the Yang, Qian & Zou (2018) article in the *Journal of Business & Economic Statistics*; corrected.
28. The Personal (2024) had a malformed DOI; replaced with the article URL.
29. Added references for every new method and for the dataset author's paper. See [REFERENCES.md](REFERENCES.md).

### Repository hygiene
30. Removed a local Windows user path, notebook outputs, kernel metadata and `.ipynb_checkpoints`. No emails, API keys or tokens were found.
31. Not included: the course instructions PDF (instructor's material), PowerPoint templates, unrelated draft notebooks and datasets (diamonds, train/test CSVs), and the dataset author's paper (cited instead). From each member's working notebooks, only the latest version is included (Alvin's random forest, Bill's MLR v1.3, Judy's EDA); earlier drafts are left out.
32. The raw Mendeley files are included as lossless parquet copies (11 MB instead of 80 MB of CSV), with attribution, so the analysis can be rebuilt without a manual download.
