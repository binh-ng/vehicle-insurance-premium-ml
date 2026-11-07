# Predicting Vehicle Insurance Premiums with Machine Learning

**Binh Nguyen (Bill) · Hui Tsz Chuen (Alvin) · Natcha Kurupakorn (Judy)**
Master of Data Science and Analytics, University of Calgary · DATA 611 Group 6 · Showcased at **DataCon 2025**

[**▶ Live demo (Streamlit)**](STREAMLIT_URL) · [Report](docs/report.pdf) · [Slides](docs/presentation.pdf) · [Poster](docs/datacon2025_poster.pdf) · [What changed](CHANGES.md) · [References](REFERENCES.md)

---

## The question

How much should a vehicle's insurance cost, and what drives that price? We used about **800,000 motor
policies (2011 to 2018) from the Ethiopian Insurance Corporation**, published openly on Mendeley Data
(Terefe, 2023a), to find the main premium drivers and to build models that predict the premium.

This repository contains our original DataCon 2025 work, unchanged, and an expanded version built
afterwards. The expanded version uses the full dataset, fixes issues we found on review, adds ten more
models, and comes with an interactive demo.

## Team

| Member | Contribution |
|---|---|
| **Binh Nguyen (Bill)** | Data cleaning, multiple linear regression, assumption testing, Lasso/Ridge/Elastic Net; expanded pipeline, GitHub showcase and Streamlit demo |
| **Hui Tsz Chuen (Alvin)** | Random forest modelling, hyperparameter tuning and feature importance |
| **Natcha Kurupakorn (Judy)** | Exploratory data analysis and interactive Plotly visualisations |

The report, slides and poster were joint group work.

## Key findings

**Test set:** 160,428 policies on vehicles the models never saw in training (20% of vehicles). OLS, Lasso and Elastic Net score within 0.03 of Ridge and are listed in the app.

| Model | R² (premium) | R² (log premium) | Median % error | Within 20% | MAE (ETB) |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.865 | 0.950 | 10.4% | 71% | 1,440 |
| LightGBM | 0.867 | 0.949 | 11.2% | 70% | 1,479 |
| Ensemble (LightGBM + XGBoost) | 0.867 | 0.950 | 10.7% | 71% | 1,449 |
| Random Forest | 0.862 | 0.948 | 9.8% | 72% | 1,454 |
| HistGradientBoosting | 0.863 | 0.948 | 10.7% | 70% | 1,475 |
| Neural Network (MLP) | 0.814 | 0.928 | 14.6% | 62% | 1,858 |
| Decision Tree | 0.818 | 0.929 | 14.2% | 62% | 1,826 |
| Ridge (log) | 0.565 | 0.880 | 28.6% | 35% | 2,710 |
| Gamma GLM (log link) | 0.501 | 0.876 | 25.3% | 38% | 2,730 |
| Tariff lookup (baseline) | 0.507 | 0.755 | 28.4% | 42% | 3,532 |

**For comparison, at DataCon 2025:** tuned random forest test R² 0.742, Ridge MLR 0.422 (different data subsets and split, so not directly comparable).

1. **Almost half of the policies are liability-only cover.** An insured value of 0 is not missing data.
   Liability (third-party) cover is mandatory in Ethiopia and priced on a nationally fixed amount per car type
   (Terefe, 2023b), so these policies need their own treatment. In the original work they were filled in
   with a median vehicle value. Keeping the zeros and adding a coverage flag cuts Ridge's median error from
   52% to 29% on the same test vehicles.
2. **For comprehensive cover, the premium is roughly a rate times the insured value** (median rate about 1.3%).
   Vehicle type, usage, engine size and age move that rate.
3. **Tree ensembles are the most accurate, and the top four are practically tied.** XGBoost, LightGBM, the tuned
   random forest and HistGradientBoosting all explain about 95% of the variation in log premium. Trained on
   policies before July 2017 and tested on later ones, the boosted models still reach a median error of 12 to 13%.
   The Gamma GLM, the standard actuarial pricing model (Goldburd et al., 2020), is less accurate but much easier to explain.
4. **Honest evaluation matters.** The same vehicle is renewed across years. All expanded models are tested on
   vehicles they never saw in training, and also on a later time period.

See [CHANGES.md](CHANGES.md) for every revision and the reasoning behind it.

## Interactive demo

The Streamlit app (`streamlit_app.py`) has seven pages:

| Page | What you can do |
|---|---|
| Overview | Project summary and headline numbers |
| Explore the data | Filter by year, coverage, vehicle type and usage; interactive Plotly charts |
| Coverage insight | Why zero insured value means liability-only cover |
| Model comparison | All models, the original DataCon results, robustness checks, predicted vs actual |
| What drives premiums | SHAP explanations of the LightGBM model |
| Premium estimator | Describe a vehicle and get an estimate with an 80% range and a "why" chart |
| Changes & references | Team, revisions since DataCon, sources |

Run it locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Reproduce the analysis

```bash
pip install -r requirements-dev.txt
python scripts/build_dataset.py        # raw files -> data/processed/policies.parquet
python scripts/train_models.py         # tune and compare all models (~1 h on 2 CPUs; --fast for a smoke test)
python scripts/replicate_original.py   # re-run the original tuned random forest
```

The two raw Mendeley files are included as lossless parquet copies in `data/raw/`. To start from the original
CSVs instead, download them from the [Mendeley page](https://data.mendeley.com/datasets/34nfrk36dt/1) into
`data/raw/`; the code uses the CSVs when they are present.

## Repository layout

```
├── streamlit_app.py            # app entry point
├── app/                        # app pages
├── src/vip/                    # data cleaning, features, model helpers
├── scripts/                    # build dataset, train models, replicate original results
├── data/raw/                   # Mendeley raw data (parquet copies, CC BY 4.0)
├── data/processed/             # cleaned policies used by the app
├── results/                    # model comparison, robustness checks, SHAP values
├── models/                     # LightGBM point and quantile models for the estimator
├── original_work/notebooks/    # DataCon 2025 notebooks (outputs cleared)
└── docs/                       # report, slides, poster
```

## Data and citation

Terefe, E. (2023). *Vehicle Insurance data* (Version 1) [Data set]. Mendeley Data.
https://doi.org/10.17632/34nfrk36dt.1. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data was obtained from the Ethiopian Insurance
Corporation. We cleaned it and derived new columns; see [data/README.md](data/README.md) for the changes.

Amounts are shown in the dataset's own units. The file does not name a currency. The values (for example, an
insured value of 519,755 for a 2007 pickup) and the dataset author's paper point to Ethiopian Birr, so the app
labels them ETB.

To cite this project, see [CITATION.cff](CITATION.cff). Full references are in [REFERENCES.md](REFERENCES.md).

## License

Code: MIT ([LICENSE](LICENSE)). Data: CC BY 4.0, © the original author. The estimator is a teaching demo
built on 2011 to 2018 data and is not an insurance quote.
