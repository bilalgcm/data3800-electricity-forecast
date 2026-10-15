# Forecasting electricity consumption in Oslo (NO1)

DATA3800 Introduction to Data Science with Scripting, OsloMet, autumn 2026. Group project.

**Research question:** How well can we forecast hourly electricity consumption in the Oslo area (price zone NO1) for the next day, and how much do temperature, time of day and calendar effects explain?

## Data

| Source | What we use | Access |
| --- | --- | --- |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Actual total load for bidding zone NO1 | Free security token (register, then email transparency@entsoe.eu to request API access) |
| [Frost API, met.no](https://frost.met.no/) | Hourly air temperature, Oslo Blindern (SN18700) | Free client ID |

## Pipeline

Each step is one script in `src/` and reads the previous step's output, so the whole analysis can be rerun from raw data.

| Step | Script | Output | Owner |
| --- | --- | --- | --- |
| 1. Fetch consumption | `src/fetch_consumption.py` | `data/raw/consumption_no1.csv` | Person 1 |
| 2. Fetch weather | `src/fetch_weather.py` | `data/raw/temperature_blindern.csv` | Person 2 |
| 3. Clean and merge | `src/build_dataset.py` | `data/processed/dataset.csv` | Person 2 |
| 4. Exploratory analysis | `notebooks/01_eda.ipynb` | `figures/` | Person 3 |
| 5. Models and evaluation | `src/models.py` | results table + figures | Person 4 |

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
python src/models.py
```

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
