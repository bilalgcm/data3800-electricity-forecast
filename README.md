# Forecasting electricity consumption in Oslo (NO1)

DATA3800 Introduction to Data Science with Scripting, OsloMet, autumn 2026. Group project.

**Research question:** How well can we forecast hourly electricity consumption in the Oslo area (price zone NO1) for the next day, and how much do temperature, time of day and calendar effects explain?

## Data

| Source | What we use | Access |
| --- | --- | --- |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Actual total load and the official day-ahead load forecast for bidding zone NO1 (the forecast is used only as a benchmark) | Free security token (register, then email transparency@entsoe.eu to request API access) |
| [Frost API, met.no](https://frost.met.no/) | Hourly air temperature, Oslo Blindern (SN18700) | Free client ID |

## Pipeline

Each step is one script in `src/` and reads the previous step's output, so the whole analysis can be rerun from raw data.

| Step | Script | Output | Owner |
| --- | --- | --- | --- |
| 1. Fetch consumption | `src/fetch_consumption.py` | `data/raw/consumption_no1.csv` | Person 1 |
| 2. Fetch weather | `src/fetch_weather.py` | `data/raw/temperature_blindern.csv` | Person 2 |
| 3. Quality check, clean and merge | `src/build_dataset.py` | `data/processed/dataset.csv`, `figures/data_quality.csv` | Person 2 |
| 4. Exploratory analysis | `src/eda.py` | `figures/eda_*.png`, `figures/eda_summary.csv` | Person 3 |
| 5. Models: tuning on a validation year, final scores | `src/models.py` | `figures/results.csv`, `figures/tuning.csv`, `figures/model_*.png`, `data/processed/predictions_test.csv` | Person 4 |
| 6. Evaluation: residuals, error by month, confusion matrix | `src/evaluation.py` | `figures/eval_*.png`, `figures/eval_*.csv` | Person 4 |
| 7. Experiments: ablation, weekend fix, temperature forecast error | `src/experiments.py` | `figures/ablation.*`, `figures/weekend_fix.*`, `figures/temperature_error.*` | Person 4 |
| 8. Final model for the agent and the service | `src/train_final_model.py` | `models/rf_final.joblib` | Person 4 |

`src/features.py` builds the calendar features for training, the agent and the service, so live data is prepared exactly like the training data. `src/plot_style.py` holds the shared figure style.

## Notebooks

The same pipeline as runnable notebooks, with the reasoning and a short interpretation under every figure and table. They import the functions in `src/`, run top to bottom, and print the Python and library versions in the first cell.

| Notebook | Content |
| --- | --- |
| `notebooks/01_data_preparation.ipynb` | Raw data, quality check, cleaning, features |
| `notebooks/02_exploratory_analysis.ipynb` | Distributions, patterns over time, correlation |
| `notebooks/03_models_and_evaluation.ipynb` | Split, tuning, model comparison, residuals, confusion matrix, experiments |
| `notebooks/04_agent.ipynb` | The agent, its tools and a live dialogue |

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # about 5 minutes
```

Notebooks 1 to 3 need no API keys, because the raw data is in `data/raw/`. Notebook 4 skips its live cells if the keys are missing.

## Setup

```bash
git clone <repo-url>
cd data3800-electricity-forecast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your own keys
```

## Run

```bash
python src/fetch_consumption.py    # needs ENTSOE_API_KEY, skip if data/raw/ is already there
python src/fetch_weather.py        # needs FROST_CLIENT_ID, skip if data/raw/ is already there
python src/build_dataset.py
python src/eda.py
python src/models.py
python src/evaluation.py
python src/experiments.py
python src/train_final_model.py
```

Without the two fetch steps the run takes about four minutes. Developed with Python 3.14; the exact library versions are pinned in `requirements.txt`.

## Forecasting agent

`agent/` contains an LLM agent that answers questions about NO1 consumption and makes real day-ahead forecasts with our model. It calls two Python tools (`agent/tools.py`): `get_forecast` (our random forest + met.no's temperature forecast + recent load from ENTSO-E) and `get_actual_load`.

```bash
python src/train_final_model.py   # trains on all data, saves models/rf_final.joblib
python agent/chat.py              # needs the AZURE_OPENAI_* keys in .env
```

Every test question and answer is recorded in `agent/test_log.md`.

## Prediction service

`service/app.py` puts the saved model behind a web address with FastAPI. `GET /health` reports which model is loaded, `POST /predict` takes one hour and returns the forecast. Input outside sensible bounds is rejected with a 422 before it reaches the model.

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8000      # then open http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"time_local": "2026-12-01T18:00", "temp_c": -3.5, "load_lag_48h": 5600, "load_lag_168h": 5450}'
```

The same service in a container, so it runs the same way on any machine:

```bash
docker build -t no1-load-forecast:1.0 .
docker run --rm -p 8000:8000 no1-load-forecast:1.0
```

## Results

Settings are chosen on the validation year 2025 (models fitted on 2023 to 2024). The final scores are on the test period January to September 2026 (models refitted on 2023 to 2025).

| Model | MAE validation (MW) | MAE test (MW) | RMSE test (MW) | R² test |
| --- | ---: | ---: | ---: | ---: |
| Baseline: same hour last week | 364.0 | 298.7 | 459.2 | 0.903 |
| ENTSO-E official day-ahead forecast | 103.8 | 150.0 | 213.5 | 0.979 |
| Linear regression | 189.3 | 185.0 | 242.2 | 0.973 |
| Decision tree (max depth 10) | 218.2 | 172.4 | 243.0 | 0.973 |
| k nearest neighbours (k = 35) | 202.4 | 183.2 | 263.6 | 0.968 |
| **Random forest** | **167.4** | **130.4** | **181.6** | **0.985** |

The random forest is the best of our models and more than halves the baseline error. Against ENTSO-E the picture depends on the period: we have the lower error on the 2026 test period, ENTSO-E is clearly better on the 2025 validation year, and our 2026 advantage comes from January to April, when ENTSO-E's error was two to three times its usual level (`figures/eval_error_by_month.csv`).

High-load hours (above 5,722 MW, the top 10 % of 2023 to 2025) treated as a yes/no warning:

| Forecast | Precision | Recall | F1 | Missed hours | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ENTSO-E | 0.911 | 0.985 | 0.947 | 17 | 109 |
| Random forest | 0.951 | 0.958 | 0.954 | 48 | 56 |

### Figures

| Figure | What it shows |
| --- | --- |
| `eda_load_over_time.png` | Daily load 2023 to 2026, with the test period marked |
| `eda_load_distribution.png` | Histogram of hourly load, with mean and median |
| `eda_load_by_hour.png`, `eda_load_box_by_hour.png` | Average daily profile, and the spread within each hour |
| `eda_load_by_weekday.png`, `eda_load_by_month.png` | Weekly and seasonal pattern |
| `eda_load_hour_weekday.png` | Hour and day of week in one heatmap |
| `eda_load_vs_temperature.png`, `eda_load_vs_temperature_hourly.png` | Load against temperature, daily and hourly |
| `eda_correlation.png` | Correlation matrix for load and the features |
| `model_tuning.png` | Training against validation error for each setting (overfitting check) |
| `model_forecast_vs_actual_week.png` | One test week: actual load, our random forest and ENTSO-E |
| `model_feature_importance.png` | Which features the random forest relies on |
| `eval_residuals.png` | Predicted against actual, and the distribution of the errors |
| `eval_error_by_month_and_hour.png` | Where the errors fall, random forest against ENTSO-E |
| `eval_confusion_matrix.png` | High-load hours: confusion matrix for the random forest and ENTSO-E |
| `ablation.png`, `weekend_fix.png`, `temperature_error.png` | The three experiments |

## Limitations

- **Temperature is measured, not forecast.** We use the actual temperature at Blindern, which acts like a perfect weather forecast. `src/experiments.py` simulates a temperature forecast error: with an error of 1 °C the MAE rises from 130 to about 140 MW, with 2 °C to about 164 MW, which is above ENTSO-E's 150 MW. The comparison with ENTSO-E is therefore not fair to ENTSO-E.
- **One test period.** The ranking against ENTSO-E changes between 2025 and 2026, so one period is not enough to say which forecast is better.
- **One weather station.** Blindern represents all of NO1, which covers a much larger area than Oslo.
- **Feature importance is impurity-based.** It can overstate features that are correlated with each other (the lagged load already contains most of the temperature effect), so we read it as a ranking and rely on the ablation test for how much each group is worth.

## Rules for the group

- Never commit `.env` or API keys. `.gitignore` already blocks them.
- Raw data is not committed to GitHub. Everyone fetches it with the scripts. The zip we deliver includes `data/raw/`.
- Work on your own branch (`git checkout -b <name>/<task>`) and open a pull request into `main`. The lead reviews before merging.
- Time is handled in UTC inside the pipeline and converted to Europe/Oslo only for analysis and plots.
- Train on older data, test on a later period. Never shuffle across time.

## Deadlines

- 18 Oct: raw data fetched and saved
- 25 Oct: Mandatory 2, working pipeline
- 2 Dec 14:00 on the exam page (the course overview says 12:00, so we deliver before 12:00): final report (PDF) + notebooks, code and data (zip), in WISEflow as Group 8
- 10 Dec 16:20 to 16:40: presentation, room N040.108