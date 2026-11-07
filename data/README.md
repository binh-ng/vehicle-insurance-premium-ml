# Data

**Source:** Terefe, E. (2023). *Vehicle Insurance data* (Version 1) [Data set]. Mendeley Data.
https://doi.org/10.17632/34nfrk36dt.1, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The data was obtained from the Ethiopian Insurance Corporation and covers motor policies from July 2011 to June 2018.

## Files

| Path | In git? | Description |
|---|---|---|
| `raw/motor_data11-14lats.parquet` | Yes | Lossless parquet copy of the raw Mendeley CSV (293,537 rows, unchanged) |
| `raw/motor_data14-2018.parquet` | Yes | Lossless parquet copy of the raw Mendeley CSV (508,499 rows, unchanged) |
| `raw/*.csv` | No | Original CSVs, if you download them (used in preference to the parquet copies) |
| `processed/policies.parquet` | Yes | Cleaned, feature-engineered policies (801,609 rows) built by `scripts/build_dataset.py` |

The raw files are redistributed under CC BY 4.0 as parquet only to keep the repository small; the values are identical to the CSVs. To rebuild the processed data, run:

```bash
python scripts/build_dataset.py
```

## Changes made to the data (CC BY 4.0 attribution)

`policies.parquet` is an adapted version of the original data: duplicates and rows without a positive premium,
production year or start date were removed; codes were mapped to labels; and the features below were derived.

| Column | Meaning |
|---|---|
| `premium` | Premium amount (target). Currency not stated in the file; most likely Ethiopian Birr (Terefe, 2023b) |
| `insured_value` | Insured value of the vehicle. 0 means liability-only cover |
| `coverage` | `Comprehensive` if insured value > 0, otherwise `Liability only` |
| `log_insured_value` | log(1 + insured value) |
| `sex` | Legal entity, Male or Female (policyholder type) |
| `insurance_type` | Private (1201), Commercial (1202), Motor trade road risk (1204) |
| `vehicle_type`, `usage`, `make` | Vehicle category, usage class, manufacturer (top 30 kept, rest "Other") |
| `vehicle_age` | Policy start year minus production year |
| `seats_num`, `carrying_capacity`, `ccm_ton` | Seats, load capacity, engine size or tonnage (clipped to plausible ranges) |
| `duration_days` | Policy length in days |
| `season` | Season of the policy start date |
| `prior_policies` | Number of earlier policies on the same vehicle in the data |
| `prior_claim` | 1 if the vehicle's previous policy had a claim paid |
| `customer_tenure` | Years since the `EFFECTIVE_YR` recorded for the policy |
| `claim_paid` | Claim paid during this policy. **Not used as a predictor** (unknown at quote time) |
