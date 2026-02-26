# HeatCash — Parametric Heat Insurance Product

In-house parametric heat insurance product for Nigeria, replicating and improving on IBISA's ClimaCash product. Built to avoid third-party fees by owning the full pipeline from raw ERA5 weather data through actuarial pricing outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Extract IBISA reference data (no API key needed)
python src/07_reverse_engineer_ibisa.py --extract

# Step 2: Download ERA5 data (requires CDS API key)
python src/01_download_era5.py --mode yearly

# Step 3: Extract temperatures per location
python src/02_extract_temperatures.py

# Step 4: Calculate heat days
python src/03_calculate_heat_days.py

# Step 5: Calculate payouts
python src/04_calculate_payouts.py

# Step 6: Compute risk metrics
python src/05_risk_metrics.py --extended

# Step 7: Calibrate thresholds
python src/06_threshold_calibration.py --all

# Step 8: Compare with IBISA
python src/07_reverse_engineer_ibisa.py --compare

# Step 9: Actuarial comparison
python src/08_actuarial_comparison.py

# Run tests
pytest tests/ -v
```

## ERA5 API Setup

1. Create a free account at https://cds.climate.copernicus.eu/
2. Get your API key from your profile page
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_UID:YOUR_API_KEY
   ```

## Project Structure

```
heat-product/
├── config/                  # Product configuration
│   ├── locations.csv        # 173 village locations (name, lat, lon)
│   ├── thresholds.csv       # Location-specific thresholds for 3 premium levels
│   └── product_config.yaml  # Product parameters
├── data/
│   ├── raw/                 # ERA5 NetCDF downloads
│   ├── processed/           # Pipeline outputs (CSVs)
│   └── ibisa_extracted/     # Extracted IBISA reference data
├── src/
│   ├── 01_download_era5.py          # ERA5 data download
│   ├── 02_extract_temperatures.py   # NetCDF → per-location CSVs
│   ├── 03_calculate_heat_days.py    # Count heat days per year
│   ├── 04_calculate_payouts.py      # Compute payouts from heat days
│   ├── 05_risk_metrics.py           # Trailing window averages
│   ├── 06_threshold_calibration.py  # Threshold analysis & calibration
│   ├── 07_reverse_engineer_ibisa.py # IBISA data extraction & comparison
│   ├── 08_actuarial_comparison.py   # Actuarial methods comparison
│   └── utils.py                     # Shared utilities
├── notebooks/               # Analysis notebooks
├── outputs/                 # Reports and figures
├── tests/                   # Unit tests
└── IBISA data/              # Original IBISA Excel files
```

## Product Parameters

| Parameter | Value |
|-----------|-------|
| Data source | ERA5 daily max 2m temperature |
| Resolution | 0.25° grid |
| Coverage period | March 1 – May 31 |
| Historical period | 1994–2024 (31 years) |
| Locations | 173 villages in Niger State, Nigeria |
| Payout | $10 per heat day |
| Strike | 1 day |
| Counting method | Total (all days above threshold) |

### Premium Options

| Option | Max Payout Days | Max Payout | Target Premium Rate |
|--------|----------------|------------|-------------------|
| 1 | 5 | $50 | ~5% |
| 2 | 7 | $70 | ~7% |
| 3 | 10 | $100 | ~10% |

## Pipeline Overview

1. **Download** ERA5 reanalysis data (daily max 2m temperature)
2. **Extract** per-location temperatures by snapping GPS to nearest 0.25° grid point
3. **Count** heat days (days exceeding location-specific threshold) per year
4. **Calculate** payouts ($10/day, capped by option max)
5. **Compute** risk metrics (trailing averages = pure premium)
6. **Calibrate** thresholds to target premium rates
7. **Validate** against IBISA reference data
8. **Compare** parametric pricing vs traditional actuarial methods
