import os
from dotenv import load_dotenv

load_dotenv()

RH_USERNAME = os.getenv("RH_USERNAME", "")
RH_PASSWORD = os.getenv("RH_PASSWORD", "")
RH_MFA_SECRET = os.getenv("RH_MFA_SECRET", "")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# Screener config
MIN_MARKET_CAP = 500_000_000       # $500M minimum
MIN_AVG_VOLUME = 500_000           # 500K shares/day
MIN_PRICE = 5.0                    # No penny stocks
MAX_PRICE = 5000.0

# Pattern matching thresholds
SIMILARITY_TOP_N = 10              # Top N recommendations
MIN_WIN_RATE_TO_LEARN = 0.40       # Only learn from categories w/ enough data
MIN_SAMPLES_TO_LEARN = 3

# Sectors to scan for recommendations
SECTORS_OF_INTEREST = [
    "Technology", "Healthcare", "Consumer Cyclical",
    "Financial Services", "Communication Services",
    "Industrials", "Basic Materials", "Energy",
    "Real Estate", "Consumer Defensive", "Utilities",
]
