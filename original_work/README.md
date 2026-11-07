# Original DataCon 2025 work

These are the notebooks we submitted for DATA 611 and presented at DataCon 2025, kept as they were apart from
clearing outputs and metadata. See [`../CHANGES.md`](../CHANGES.md) for the issues we found and fixed later.

| Notebook | Author | Content |
|---|---|---|
| `01_group6_final_notebook.ipynb` | Group 6 | Final combined notebook: data wrangling, EDA, random forest, MLR |
| `02_random_forest_alvin_v1.0.ipynb` | Hui Tsz Chuen (Alvin) | Random forest baseline, tuning, feature importance |
| `03_multiple_linear_regression_binh_v1.3.ipynb` | Binh Nguyen (Bill) | Data cleaning, MLR, assumption tests, Cook's distance, Lasso/Ridge/Elastic Net |
| `04_eda_visualisation_judy_v0.1.ipynb` | Natcha Kurupakorn (Judy) | Exploratory data analysis with interactive Plotly charts |

The notebooks read `motor_data14-2018.csv` from the working directory. Download it from the Mendeley page or load the parquet copy in `data/raw/` with `pd.read_parquet`.

`scripts/replicate_original.py` re-runs the tuned random forest from notebook 01 and checks the reported scores.
The report, slides and poster are in [`../docs`](../docs).
