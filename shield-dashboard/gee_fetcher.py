"""
Google Earth Engine Data Fetcher for SHIELD Heat Dashboard.

Fetches current-season ERA5-Land daily maximum temperatures for 173 villages
in Niger State, Nigeria, computes heat days against village-specific thresholds,
and saves the results to live_data.csv.

Data source: ECMWF ERA5-Land Daily Aggregated (ECMWF/ERA5_LAND/DAILY_AGGR)
Band: temperature_2m_max (Kelvin, converted to Celsius)
Resolution: ~9km (0.1 degree)

Coverage period: March 1 to May 31 of the current year (92 days).
ERA5-Land has approximately 5 days of latency, so the effective end date
is min(today - 5 days, May 31).
"""

import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
LOCATIONS_CSV = CONFIG_DIR / "locations.csv"
THRESHOLDS_CSV = CONFIG_DIR / "thresholds.csv"
OUTPUT_CSV = SCRIPT_DIR / "live_data.csv"

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
TEMP_BAND = "temperature_2m_max"
KELVIN_OFFSET = 273.15

SEASON_START_MONTH = 3   # March
SEASON_START_DAY = 1
SEASON_END_MONTH = 5     # May
SEASON_END_DAY = 31
TOTAL_SEASON_DAYS = 92

ERA5_LATENCY_DAYS = 5    # ERA5-Land data lags ~5 days behind real-time

MAX_PAYOUT_DAYS = 7      # Payout is capped at 7 heat days
PAYOUT_PER_DAY = 10      # $10 per heat day

# Batch size for GEE getInfo calls (number of images per batch).
# Keeps individual requests small enough to avoid GEE timeout / memory limits.
GEE_IMAGE_BATCH_SIZE = 25


# ---------------------------------------------------------------------------
# GEE Initialization
# ---------------------------------------------------------------------------

def initialize_gee(project_id=None):
    """Initialize the Google Earth Engine API.

    Attempts authentication using default credentials first. If a *project_id*
    is supplied it is passed to ``ee.Initialize`` (required for Cloud projects).

    Raises
    ------
    ImportError
        If the ``ee`` package is not installed.
    RuntimeError
        If authentication / initialisation fails.
    """
    try:
        import ee  # noqa: F811
    except ImportError:
        raise ImportError(
            "The 'earthengine-api' package is not installed.\n"
            "Install it with:  pip install earthengine-api\n"
            "Then authenticate with:  earthengine authenticate"
        )

    try:
        init_kwargs = {}
        if project_id:
            init_kwargs["project"] = project_id

        # Try initializing with existing credentials first.
        ee.Initialize(**init_kwargs)
        print("[GEE] Initialized successfully.")
    except Exception as init_err:
        # If that fails, attempt to authenticate then re-initialise.
        try:
            print("[GEE] Default credentials not found — attempting authentication...")
            ee.Authenticate()
            ee.Initialize(**init_kwargs)
            print("[GEE] Authenticated and initialized successfully.")
        except Exception as auth_err:
            raise RuntimeError(
                "Could not authenticate with Google Earth Engine.\n"
                "Run 'earthengine authenticate' in your terminal and try again.\n"
                f"Original error: {auth_err}"
            ) from init_err


# ---------------------------------------------------------------------------
# Temperature Fetching
# ---------------------------------------------------------------------------

def _build_feature_collection(locations_df):
    """Convert a locations DataFrame to an ee.FeatureCollection of points.

    Each feature carries the village *name* as a property so results can be
    joined back to the input table.

    Parameters
    ----------
    locations_df : pd.DataFrame
        Must contain columns ``name``, ``latitude``, ``longitude``.

    Returns
    -------
    ee.FeatureCollection
    """
    import ee

    features = []
    for _, row in locations_df.iterrows():
        point = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        feat = ee.Feature(point, {"name": row["name"]})
        features.append(feat)
    return ee.FeatureCollection(features)


def _extract_temps_for_image(image, villages_fc):
    """Extract the max temperature value at every village point for one image.

    Uses ``reduceRegions`` with ``ee.Reducer.first()`` (single-pixel sampling)
    which is efficient for point geometries.

    Returns an ee.FeatureCollection where each feature has properties:
    ``name``, ``date``, and ``max_temp_c``.
    """
    import ee

    # Get the image date as a string (YYYY-MM-dd)
    date_str = image.date().format("YYYY-MM-dd")

    reduced = image.select(TEMP_BAND).reduceRegions(
        collection=villages_fc,
        reducer=ee.Reducer.first(),
        scale=11132,  # ~0.1 degree at the equator ≈ 11 132 m
    )

    def _add_date_and_convert(feature):
        temp_k = ee.Number(feature.get("first"))
        # Guard against null pixels — use -9999 as sentinel so we can filter later.
        temp_c = ee.Algorithms.If(
            temp_k,
            temp_k.subtract(KELVIN_OFFSET),
            -9999,
        )
        return feature.set({"date": date_str, "max_temp_c": temp_c})

    return reduced.map(_add_date_and_convert)


def fetch_season_temperatures(locations_df, start_date, end_date):
    """Fetch daily maximum 2 m temperatures for all villages from ERA5-Land.

    Processes the ERA5-Land ImageCollection in batches using
    ``reduceRegions`` so that all 173 villages are sampled in a single
    server-side operation per image.

    Parameters
    ----------
    locations_df : pd.DataFrame
        Village locations with columns ``name``, ``latitude``, ``longitude``.
    start_date : str
        Start date inclusive, ``"YYYY-MM-DD"`` format.
    end_date : str
        End date inclusive, ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        Columns: ``name``, ``date``, ``max_temp_c``.
        One row per village per day.  Rows where the pixel had no data are
        excluded.
    """
    import ee

    print(f"[GEE] Fetching ERA5-Land temperatures from {start_date} to {end_date} ...")

    villages_fc = _build_feature_collection(locations_df)

    # Build the image collection for the date range.
    # GEE filterDate end is exclusive, so add one day.
    end_exclusive = (
        datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
    ).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start_date, end_exclusive)
        .select(TEMP_BAND)
    )

    # Get the list of image IDs so we can batch them.
    image_ids = collection.aggregate_array("system:index").getInfo()
    total_images = len(image_ids)
    print(f"[GEE] Found {total_images} daily images in the date range.")

    if total_images == 0:
        print("[GEE] No images available for the requested date range.")
        return pd.DataFrame(columns=["name", "date", "max_temp_c"])

    # Process in batches to avoid exceeding GEE computation limits.
    all_records = []
    for batch_start in range(0, total_images, GEE_IMAGE_BATCH_SIZE):
        batch_end = min(batch_start + GEE_IMAGE_BATCH_SIZE, total_images)
        batch_ids = image_ids[batch_start:batch_end]
        print(
            f"[GEE] Processing images {batch_start + 1}–{batch_end} "
            f"of {total_images} ..."
        )

        # Build a merged FeatureCollection for this batch.
        batch_features = ee.FeatureCollection([])
        for img_id in batch_ids:
            image = ee.Image(f"{ERA5_COLLECTION}/{img_id}")
            extracted = _extract_temps_for_image(image, villages_fc)
            batch_features = batch_features.merge(extracted)

        # Pull results to the client.
        try:
            result = batch_features.getInfo()
        except Exception as exc:
            print(f"[GEE] Warning — batch request failed: {exc}")
            continue

        for feat in result.get("features", []):
            props = feat.get("properties", {})
            temp_c = props.get("max_temp_c")
            if temp_c is not None and temp_c != -9999:
                all_records.append(
                    {
                        "name": props.get("name"),
                        "date": props.get("date"),
                        "max_temp_c": round(temp_c, 2),
                    }
                )

    print(f"[GEE] Retrieved {len(all_records)} village-day temperature records.")
    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Heat-Day & Payout Computation
# ---------------------------------------------------------------------------

def compute_heat_days(temp_df, thresholds_df):
    """Count heat days per village and compute payouts.

    A *heat day* for a village is any day where the daily maximum temperature
    meets or exceeds the village's threshold (``threshold_7pct`` column).

    Payout logic:
        payout = min(heat_days, 7) * $10   (capped at $70)

    Status labels:
        - ``"Safe"``      — 0 heat days
        - ``"Triggered"`` — 1 to 6 heat days
        - ``"Capped"``    — 7 or more heat days

    Parameters
    ----------
    temp_df : pd.DataFrame
        Daily temperature records with columns ``name``, ``date``,
        ``max_temp_c``.
    thresholds_df : pd.DataFrame
        Village thresholds with columns ``name`` and ``threshold_7pct``.

    Returns
    -------
    pd.DataFrame
        One row per village with columns: ``name``, ``threshold``,
        ``heat_days``, ``payout``, ``status``.
    """
    # Merge temperatures with thresholds.
    merged = temp_df.merge(
        thresholds_df[["name", "threshold_7pct"]],
        on="name",
        how="left",
    )

    # Flag heat days.
    merged["is_heat_day"] = merged["max_temp_c"] >= merged["threshold_7pct"]

    # Aggregate per village.
    heat_counts = (
        merged.groupby("name")["is_heat_day"]
        .sum()
        .astype(int)
        .reset_index()
        .rename(columns={"is_heat_day": "heat_days"})
    )

    # Bring in thresholds for villages that may have had zero records.
    result = thresholds_df[["name", "threshold_7pct"]].merge(
        heat_counts, on="name", how="left"
    )
    result["heat_days"] = result["heat_days"].fillna(0).astype(int)
    result = result.rename(columns={"threshold_7pct": "threshold"})

    # Payout: min(heat_days, 7) * 10
    result["payout"] = result["heat_days"].clip(upper=MAX_PAYOUT_DAYS) * PAYOUT_PER_DAY

    # Status label.
    def _status(days):
        if days == 0:
            return "Safe"
        elif days < MAX_PAYOUT_DAYS:
            return "Triggered"
        else:
            return "Capped"

    result["status"] = result["heat_days"].apply(_status)

    return result


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def fetch_live_data(project_id=None):
    """Orchestrate the full data-fetch pipeline.

    Steps:
        1. Initialise Google Earth Engine.
        2. Load village locations and thresholds from the config directory.
        3. Determine the date range for the current season (Mar 1 – May 31),
           capped at *today minus 5 days* to account for ERA5 latency.
        4. Fetch daily maximum temperatures from ERA5-Land via GEE.
        5. Compute heat days and payouts.
        6. Save results to ``live_data.csv``.

    Parameters
    ----------
    project_id : str, optional
        Google Cloud project ID for GEE. If *None*, default credentials are
        used.

    Returns
    -------
    pd.DataFrame
        The results DataFrame that was saved to ``live_data.csv``.
    """
    # ------------------------------------------------------------------
    # 1. Initialise GEE
    # ------------------------------------------------------------------
    initialize_gee(project_id=project_id)

    # ------------------------------------------------------------------
    # 2. Load config files
    # ------------------------------------------------------------------
    print(f"[CONFIG] Loading locations from {LOCATIONS_CSV}")
    locations_df = pd.read_csv(LOCATIONS_CSV)
    print(f"[CONFIG] Loaded {len(locations_df)} village locations.")

    print(f"[CONFIG] Loading thresholds from {THRESHOLDS_CSV}")
    thresholds_df = pd.read_csv(THRESHOLDS_CSV)
    print(f"[CONFIG] Loaded {len(thresholds_df)} village thresholds.")

    # ------------------------------------------------------------------
    # 3. Determine date range
    # ------------------------------------------------------------------
    today = datetime.date.today()
    current_year = today.year

    season_start = datetime.date(current_year, SEASON_START_MONTH, SEASON_START_DAY)
    season_end = datetime.date(current_year, SEASON_END_MONTH, SEASON_END_DAY)

    # ERA5 data is available up to ~5 days ago.
    data_available_through = today - datetime.timedelta(days=ERA5_LATENCY_DAYS)

    if data_available_through < season_start:
        print(
            f"[INFO] Season has not started yet or no ERA5 data available. "
            f"Season starts {season_start}, data available through {data_available_through}."
        )
        # Return an empty results DataFrame with the correct schema.
        empty = locations_df[["name", "latitude", "longitude"]].copy()
        empty = empty.merge(
            thresholds_df[["name", "threshold_7pct"]].rename(
                columns={"threshold_7pct": "threshold"}
            ),
            on="name",
            how="left",
        )
        empty["heat_days"] = 0
        empty["payout"] = 0
        empty["status"] = "Safe"
        empty["last_updated"] = str(data_available_through)
        empty["season_day"] = 0
        empty["total_season_days"] = TOTAL_SEASON_DAYS
        empty.to_csv(OUTPUT_CSV, index=False)
        print(f"[OUTPUT] Saved empty results to {OUTPUT_CSV}")
        return empty

    effective_end = min(data_available_through, season_end)
    start_str = season_start.strftime("%Y-%m-%d")
    end_str = effective_end.strftime("%Y-%m-%d")
    season_day = (effective_end - season_start).days + 1

    print(f"[DATE] Season: {season_start} to {season_end} ({TOTAL_SEASON_DAYS} days)")
    print(f"[DATE] Fetching: {start_str} to {end_str} (day {season_day} of season)")

    # ------------------------------------------------------------------
    # 4. Fetch temperatures
    # ------------------------------------------------------------------
    temp_df = fetch_season_temperatures(locations_df, start_str, end_str)

    # ------------------------------------------------------------------
    # 5. Compute heat days and payouts
    # ------------------------------------------------------------------
    if temp_df.empty:
        print("[WARN] No temperature data returned. Generating empty results.")
        results = locations_df[["name", "latitude", "longitude"]].copy()
        results = results.merge(
            thresholds_df[["name", "threshold_7pct"]].rename(
                columns={"threshold_7pct": "threshold"}
            ),
            on="name",
            how="left",
        )
        results["heat_days"] = 0
        results["payout"] = 0
        results["status"] = "Safe"
    else:
        results = compute_heat_days(temp_df, thresholds_df)
        # Attach coordinates.
        results = results.merge(
            locations_df[["name", "latitude", "longitude"]],
            on="name",
            how="left",
        )

    # ------------------------------------------------------------------
    # 6. Add metadata columns and save
    # ------------------------------------------------------------------
    results["last_updated"] = end_str
    results["season_day"] = season_day
    results["total_season_days"] = TOTAL_SEASON_DAYS

    # Ensure column order matches the specification.
    output_columns = [
        "name",
        "latitude",
        "longitude",
        "threshold",
        "heat_days",
        "payout",
        "status",
        "last_updated",
        "season_day",
        "total_season_days",
    ]
    results = results[output_columns]

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"[OUTPUT] Saved {len(results)} rows to {OUTPUT_CSV}")

    return results


# ---------------------------------------------------------------------------
# Standalone execution for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch ERA5-Land heat data for Niger State villages via GEE."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud project ID for Earth Engine (optional).",
    )
    args = parser.parse_args()

    df = fetch_live_data(project_id=args.project)
    print("\n--- Results Summary ---")
    print(f"Total villages:  {len(df)}")
    print(f"Safe:            {(df['status'] == 'Safe').sum()}")
    print(f"Triggered:       {(df['status'] == 'Triggered').sum()}")
    print(f"Capped:          {(df['status'] == 'Capped').sum()}")
    print(f"Total payout:    ${df['payout'].sum()}")
