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
| 3. Clean and merge | `src/build_dataset.py` | `data/processed/dataset.csv` | Person 2 |
| 4. Exploratory analysis | `src/eda.py` | `figures/eda_*.png`, `figures/eda_summary.csv` | Person 3 |
| 5. Models and evaluation | `src/models.py` | `figures/results.csv`, `figures/model_*.png`, `data/processed/predictions_test.csv` | Person 4 |
| 6. Experiments (ablation test, weekend fix) | `src/experiments.py` | `figures/ablation.*`, `figures/weekend_fix.*` | Person 4 |

`src/plot_style.py` holds the shared figure style, so every plot in the report uses the same colours and fonts.

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
python src/fetch_consumption.py
python src/fetch_weather.py
python src/build_dataset.py
python src/eda.py
python src/models.py
python src/experiments.py
```

The full run takes a few minutes, most of it fetching data from the APIs.

## Forecasting agent

`agent/` contains an LLM agent that answers questions about NO1 consumption and makes real day-ahead forecasts with our model. It calls two Python tools (`agent/tools.py`): `get_forecast` (our random forest + met.no's temperature forecast + recent load from ENTSO-E) and `get_actual_load`.

```bash
python src/train_final_model.py   # trains on all data, saves models/rf_final.joblib
python agent/chat.py              # needs the AZURE_OPENAI_* keys in .env
```

Every test question and answer is recorded in `agent/test_log.md`.

## Results

Test period: January to September 2026. The models are trained on 2023 to 2025 only.

| Model | MAE (MW) | RMSE (MW) |
| --- | ---: | ---: |
| Baseline: same hour last week | 298.7 | 459.2 |
| ENTSO-E official day-ahead forecast | 150.0 | 213.5 |
| Linear regression | 185.0 | 242.2 |
| **Random forest** | **130.4** | **181.6** |

The random forest more than halves the baseline error and beats the linear model, which cannot capture the non-linear link between temperature and load.

### Figures

| Figure | What it shows |
| --- | --- |
| `eda_load_over_time.png` | Daily load 2023 to 2026, with the test period marked |
| `eda_load_by_hour.png` | Average daily profile, weekdays vs weekends |
| `eda_load_by_weekday.png` | Average load per day of week |
| `eda_load_by_month.png` | Seasonal pattern, average load per month |
| `eda_load_vs_temperature.png` | Daily load against daily temperature, with correlation |
| `model_forecast_vs_actual_week.png` | One test week: actual load, our random forest and ENTSO-E |
| `model_feature_importance.png` | Which features the random forest relies on |

## Limitations

- **Temperature is measured, not forecast.** We use the actual temperature at Blindern, which acts like a perfect weather forecast. A real day-ahead forecast would use a temperature forecast, so our errors are somewhat optimistic. This is the main reason the comparison with ENTSO-E is not fully fair.
- **One weather station.** Blindern represents all of NO1, which covers a much larger area than Oslo.
- **Feature importance is impurity-based.** It can overstate features that are correlated with each other (for example temperature and month), so we read it as a ranking, not exact shares.

## Rules for the group

- Never commit `.env` or API keys. `.gitignore` already blocks them.
- Raw data is not committed. Everyone fetches it with the scripts.
- Work on your own branch (`git checkout -b <name>/<task>`) and open a pull request into `main`. The lead reviews before merging.
- Time is handled in UTC inside the pipeline and converted to Europe/Oslo only for analysis and plots.
- Train on older data, test on a later period. Never shuffle across time.

## Deadlines

- 18 Oct: raw data fetched and saved
- 25 Oct: Mandatory 2, working pipeline
- 2 Dec 12:00: final report (PDF) + code and data (zip)