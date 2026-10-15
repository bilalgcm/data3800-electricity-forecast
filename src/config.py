"""Shared settings for the whole pipeline."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# Period to analyse (UTC). Adjust once we know how far back the data is complete.
START = "2023-01-01"
END = "2026-10-01"

# Last part of the period is held out as the test set.
TEST_START = "2026-01-01"

TZ = "Europe/Oslo"

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
FROST_CLIENT_ID = os.getenv("FROST_CLIENT_ID")

BIDDING_ZONE = "NO_1"        # entsoe-py code for NO1
FROST_STATION = "SN18700"    # Oslo Blindern
