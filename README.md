# Nepal Flood & Weather Analysis: Next-Day River Discharge Prediction

A machine-learning project that uses weather observations, historical river discharge, seasonal information, and station characteristics to predict **next-day river discharge** and classify **potential high-flow days** across 10 river stations in Nepal.

> **Project status:** Exploratory machine-learning project. The high-flow labels are based on statistical, station-specific thresholds and are **not official flood-warning levels**. The models should not be used for operational warnings without further validation.

## Project Overview

River discharge varies by station and season. This project explores the relationship between weather and river flow, creates temporal features from past rainfall and discharge, and evaluates regression and classification models using a chronological train/test split.

The project contains two modelling tasks:

1. **Regression:** estimate the next day's river discharge in cubic metres per second (m³/s).
2. **High-flow classification:** predict whether the next day's discharge will exceed the station-specific 90th-percentile threshold calculated from the training period.

## Dataset

The notebook loads the CSV file:

`CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv`

The notebook itself does not record an explicit dataset source URL. Add the original dataset link here before publishing the repository.

### Dataset summary

| Property | Details |
|---|---|
| Original records | 13,390 rows |
| Original columns | 19 |
| Date range | 1 January 2023 – 31 August 2026 |
| Locations | 10 river stations |
| Missing values in original data | 0 reported |
| Duplicate rows | 0 reported |
| Duplicate date–station pairs | 0 reported |
| Target for regression | `river_discharge_m3s` shifted to the next day |

The dataset includes date and station identifiers, river/basin information, station coordinates and elevation, weather variables, and observed river discharge.

### Main variables

- **Weather:** precipitation, rainfall, soil moisture, mean temperature, dew point, precipitation hours, relative humidity, maximum wind speed, maximum wind gusts, and dominant wind direction.
- **Station information:** location, river, basin, station identifier, latitude, longitude, and elevation.
- **River observation:** `river_discharge_m3s`.

## Project Workflow

### 1. Data understanding and quality checks

- Inspected dataset dimensions, data types, and descriptive statistics.
- Converted `date` to a datetime type.
- Checked missing values, duplicate rows, and duplicate date–station pairs.
- Examined the discharge distribution and station-level differences.

### 2. Exploratory data analysis

The notebook investigates:

- The distribution and skew of river discharge.
- Average discharge across stations.
- Rainfall and precipitation versus discharge.
- Correlations between same-day weather variables and discharge.
- Monthly and seasonal flow patterns.
- Unusually high flows using station-specific 95th-percentile thresholds.
- Differences in flow distributions across stations.

**Selected findings from the notebook:**

- The pooled monthly average discharge peaks in **August at approximately 72.46 m³/s**.
- Kusum has the highest average discharge in the dataset, at approximately **125.81 m³/s**.
- Dew point has the largest same-day linear correlation with discharge among the listed weather variables (**r ≈ 0.226**); same-day rainfall has a weaker correlation (**r ≈ 0.100**).
- Flow scales vary considerably between stations, so pooled results should be interpreted with care.

These are descriptive findings for the available dataset; correlation alone does not establish causality.

### 3. Feature engineering

Features are generated after sorting observations by station and date. The notebook creates:

- **Rainfall lags:** previous 1, 2, and 3 days.
- **Rolling rainfall:** previous 3-day rainfall sum, previous 7-day rainfall sum, and previous 7-day rainfall mean.
- **Historical discharge:** previous 1-day and 2-day discharge, plus the previous 7-day mean discharge.
- **Calendar features:** year and month.
- **Station encoding:** one-hot encoded station/location columns.

The rolling rainfall and discharge windows are shifted to use prior observations rather than the current day's value in the rolling window.

After removing rows without sufficient history and rows without a next-day target, the modelling dataset contains **13,310 rows**. The model uses 35 input features: 14 baseline features and 21 additional engineered/calendar/station features.

### 4. Target and chronological split

For each station, the target is created by shifting `river_discharge_m3s` one row forward in date order:

- **Target:** `target_next_day`
- **Target date:** `target_date`

A time-based split is used instead of a random split, to keep the test period later than the training period.

| Split | Rows | Target-date range |
|---|---:|---|
| Training | 10,630 | 9 January 2023 – 6 December 2025 |
| Testing | 2,680 | 7 December 2025 – 31 August 2026 |

The notebook includes assertions to verify that the target-date ranges do not overlap. The test set covers data through August 2026; September 2026 is not included.

### 5. Regression models

Four regressors are trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor

Evaluation metrics include MAE, MSE, RMSE, and R².

#### Regression results

| Model | MAE (m³/s) | MSE | RMSE (m³/s) | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 6.0606 | 226.3006 | **15.0433** | **0.9344** |
| Random Forest | **2.4167** | 277.3969 | 16.6552 | 0.9196 |
| Gradient Boosting | 2.8414 | 374.0599 | 19.3406 | 0.8916 |
| Decision Tree | 3.1119 | 532.8228 | 23.0830 | 0.8456 |

**Interpretation:** Linear Regression has the lowest RMSE and highest R², while Random Forest has the lowest MAE. The preferred model therefore depends on the error measure and intended use. Linear Regression can generate negative discharge predictions, which are not physically meaningful.

### 6. Feature-engineering comparison

The notebook compares a Random Forest using the baseline feature set with a Random Forest using the expanded feature set.

| Random Forest feature set | MAE (m³/s) | RMSE (m³/s) | R² |
|---|---:|---:|---:|
| Baseline features | 2.4708 | 17.8201 | 0.9080 |
| With feature engineering | **2.4167** | **16.6552** | **0.9196** |

The expanded feature set reduced MAE by **2.19%** and RMSE by **6.54%** in this experiment. This comparison evaluates the combined feature additions; it does not isolate the effect of each individual engineered feature.

The Random Forest regressor assigns its largest feature-importance score to current river discharge (approximately **0.8136**). Feature importance is model-specific and should not be interpreted as a causal contribution or as a measure of forecast accuracy.

### 7. High-flow classification

The classification target is defined as:

- **High flow = 1:** next-day discharge exceeds that station's 90th-percentile discharge threshold, calculated from the training period.
- **Normal flow = 0:** otherwise.

The training set contains approximately **10%** high-flow observations; the test set contains approximately **6.4%**. The notebook uses `class_weight="balanced"` for Logistic Regression, Decision Tree, and Random Forest to address class imbalance.

#### Classification results

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.840 | 0.287 | **1.000** | 0.446 |
| Decision Tree | 0.936 | 0.500 | 0.663 | 0.570 |
| Random Forest | **0.970** | **0.769** | 0.756 | **0.762** |

Random Forest provides the best F1 score among the three tested classifiers. Logistic Regression reaches 100% recall on the test set but has lower precision, meaning it produces more false alarms. In flood-risk applications, missed high-flow events and excessive false alarms both matter, so threshold selection should reflect the intended operational trade-off.

The station-specific 90th-percentile thresholds are **statistical labels created for this project**, not official hydrological alert thresholds.

## Tools and Libraries

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Add the dataset

Place `CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv` in the project directory, or update the CSV path in the notebook. The current notebook uses a Google Colab path:

```python
df = pd.read_csv('/content/CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv')
```

For a local run, for example, change it to:

```python
df = pd.read_csv('CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv')
```

Make sure you have permission to redistribute the dataset before committing it to a public repository. If redistribution is not permitted, provide the original dataset link and instructions for obtaining it instead.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. The notebook also exports a processed dataset named `nepal_flood_processed.csv` during execution.

## Suggested Repository Structure

```text
nepal-flood-discharge-prediction/
├── Nepal_floods_ml(5).ipynb
├── README.md
├── requirements.txt                 # optional: save dependencies here
├── CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv  # only if redistribution is permitted
└── nepal_flood_processed.csv        # generated by the notebook; optional
```

## Limitations and Future Improvements

- **No persistence benchmark:** the notebook does not compare the models against a simple forecast in which tomorrow's discharge equals today's discharge. This baseline is important for measuring whether the machine-learning model adds forecasting value.
- **Pooled metrics:** stations have very different discharge scales, and large flows from stations such as Kusum can strongly affect aggregate metrics. Report station-level metrics to understand performance more fully.
- **Limited out-of-time evaluation:** results are based on one chronological split; the test period ends on 31 August 2026 and includes only part of the 2026 monsoon season.
- **High-flow threshold is statistical:** station-specific 90th-percentile labels are not equivalent to official warning thresholds.
- **Extreme-flow errors:** the largest events can still be underpredicted or overpredicted; inspect event-level errors before any operational use.
- **Regression constraints:** Linear Regression can predict negative discharge. Consider a non-negative/transformed-target approach and evaluate the effect on metrics.
- **No reported rolling-origin validation or extensive hyperparameter search:** additional validation is required to assess robustness.
- **Feature importance is not causality:** use domain knowledge and additional analysis before drawing causal conclusions about weather and discharge.

Recommended next steps include a persistence baseline, per-station error metrics, rolling-origin validation, calibration and threshold tuning for high-flow classification, and incorporating upstream observations or rainfall forecasts where available.

## Results Summary

- **Best RMSE and R²:** Linear Regression (RMSE 15.0433 m³/s; R² 0.9344).
- **Best regression MAE:** Random Forest (MAE 2.4167 m³/s).
- **Best high-flow classifier by F1:** Random Forest (F1 0.762; precision 0.769; recall 0.756).
- **Feature engineering:** improved Random Forest MAE by 2.19% and RMSE by 6.54% compared with the baseline feature set in the notebook's experiment.

These results are encouraging for an exploratory project, but they do not establish readiness for real-world flood-warning deployment.

## Acknowledgements and Data Source

The notebook's data filename indicates a Nepal flood/weather Kaggle dataset, but an explicit source URL is not included in the notebook. Add the exact dataset URL and attribution here before publishing the repository.

## License

No software or dataset license is specified in the notebook. Choose a suitable repository license before publishing, and verify the dataset's terms separately.
