"""
One Acre Shield - Extreme Heat Household Protection Dashboard
=============================================================
Streamlit dashboard for monitoring parametric heat insurance payouts
across 173 villages in Niger State, Nigeria.

Product: 7% scenario only
  - Payout: $10 per heat day
  - Cap: 7 heat days = $70 max payout
  - Coverage: March 1 - May 31 (92 days)
  - Premium: ~$5.61
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import base64

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="One Acre Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------
KKM_GREEN = "#2a8068"
BRIGHT_GREEN = "#3cb88a"
BRIGHT_GREEN_ALT = "#4db892"
ACCENT_GREEN = "#34a07a"
NAVY = "#212558"
BG_DARK = "#141836"
CARD_BG = "#1e2150"
BORDER = "#2d3068"
WARNING = "#ff9f43"
DANGER = "#ff6b6b"
TEXT_PRIMARY = "#e0e6ed"
TEXT_SECONDARY = "#8892a4"
SAFE_GREEN = "#2a8068"

# Product parameters
PAYOUT_PER_DAY = 10
CAP_DAYS = 7
MAX_PAYOUT = CAP_DAYS * PAYOUT_PER_DAY  # $70
COVERAGE_DAYS = 92
PREMIUM = 5.61

# ---------------------------------------------------------------------------
# Resolve file paths (relative to this script's parent directory)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

LOCATIONS_PATH = BASE_DIR / "config" / "locations.csv"
THRESHOLDS_PATH = BASE_DIR / "config" / "thresholds.csv"
HEAT_DAYS_PATH = BASE_DIR / "data" / "processed" / "heat_days_7pct.csv"
LOGO_PATH = BASE_DIR / "Shield Assets" / "One Acre Shield_Logo_White.png"
LIVE_DATA_PATH = APP_DIR / "live_data.csv"

# ---------------------------------------------------------------------------
# Custom CSS — dark branded look
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    /* Global background and text */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_PRIMARY};
    }}

    /* Hide default Streamlit header/footer for cleaner look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* KPI metric cards */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        min-height: 130px;
    }}
    .kpi-label {{
        color: {TEXT_SECONDARY};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .kpi-sub {{
        color: {TEXT_SECONDARY};
        font-size: 0.82rem;
    }}

    /* Status badges */
    .badge-safe {{
        background: rgba(42, 128, 104, 0.25);
        color: {BRIGHT_GREEN};
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .badge-triggered {{
        background: rgba(255, 159, 67, 0.2);
        color: {WARNING};
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .badge-capped {{
        background: rgba(255, 107, 107, 0.2);
        color: {DANGER};
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    /* Streamlit metric override */
    [data-testid="stMetricValue"] {{
        color: {BRIGHT_GREEN};
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY};
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* Section headers */
    .section-header {{
        color: {TEXT_PRIMARY};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER};
    }}

    /* Info banner */
    .info-banner {{
        background: rgba(42, 128, 104, 0.15);
        border: 1px solid {KKM_GREEN};
        border-radius: 8px;
        padding: 16px 20px;
        color: {TEXT_PRIMARY};
        margin: 10px 0;
    }}

    /* Product info block in sidebar */
    .product-info {{
        background: rgba(42, 128, 104, 0.1);
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 12px;
        margin: 12px 0;
        font-size: 0.85rem;
    }}
    .product-info .param-label {{
        color: {TEXT_SECONDARY};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .product-info .param-value {{
        color: {BRIGHT_GREEN};
        font-weight: 600;
    }}

    /* Plotly chart container */
    .stPlotlyChart {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Selectbox / input styling */
    .stSelectbox label, .stTextInput label {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* Sidebar divider */
    .sidebar-divider {{
        border-top: 1px solid {BORDER};
        margin: 16px 0;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_locations() -> pd.DataFrame:
    """Load village locations (name, latitude, longitude)."""
    try:
        return pd.read_csv(LOCATIONS_PATH)
    except FileNotFoundError:
        st.error(f"Locations file not found: {LOCATIONS_PATH}")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])


@st.cache_data
def load_thresholds() -> pd.DataFrame:
    """Load calibrated thresholds per village (7% scenario)."""
    try:
        df = pd.read_csv(THRESHOLDS_PATH)
        return df[["name", "threshold_7pct"]].rename(
            columns={"threshold_7pct": "threshold"}
        )
    except FileNotFoundError:
        st.error(f"Thresholds file not found: {THRESHOLDS_PATH}")
        return pd.DataFrame(columns=["name", "threshold"])


@st.cache_data
def load_heat_days() -> pd.DataFrame:
    """Load historical heat day counts (year x village)."""
    try:
        return pd.read_csv(HEAT_DAYS_PATH)
    except FileNotFoundError:
        st.error(f"Heat days file not found: {HEAT_DAYS_PATH}")
        return pd.DataFrame()


def load_live_data() -> pd.DataFrame | None:
    """Load live 2026 data if it exists (not cached — may change)."""
    try:
        if LIVE_DATA_PATH.exists():
            return pd.read_csv(LIVE_DATA_PATH)
    except Exception:
        pass
    return None


def load_logo_base64() -> str | None:
    """Load the logo as a base64 string for embedding in HTML."""
    try:
        if LOGO_PATH.exists():
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def compute_payout(heat_days: int | float) -> float:
    """Compute payout: $10 per heat day, capped at 7 days ($70)."""
    return min(int(heat_days), CAP_DAYS) * PAYOUT_PER_DAY


def get_status(heat_days: int | float) -> str:
    """Classify village status based on heat day count."""
    hd = int(heat_days)
    if hd == 0:
        return "Safe"
    elif hd < CAP_DAYS:
        return "Triggered"
    else:
        return "Capped"


def get_status_color(status: str) -> str:
    """Return hex color for a status string."""
    return {
        "Safe": SAFE_GREEN,
        "Triggered": WARNING,
        "Capped": DANGER,
    }.get(status, TEXT_SECONDARY)


def build_village_table(
    heat_days_series: pd.Series,
    thresholds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full village payout table for a given year's heat day data.

    Parameters
    ----------
    heat_days_series : pd.Series
        Index = village names, values = heat day counts for one year.
    thresholds_df : pd.DataFrame
        Columns: name, threshold.

    Returns
    -------
    pd.DataFrame with columns:
        Village, Threshold (C), Heat Days, Payout ($), % of Cap, Status
    """
    df = thresholds_df.copy()
    df = df.rename(columns={"name": "Village", "threshold": "Threshold (°C)"})

    # Map heat days
    df["Heat Days"] = df["Village"].map(heat_days_series).fillna(0).astype(int)

    # Compute payouts
    df["Payout ($)"] = df["Heat Days"].apply(compute_payout)
    df["% of Cap"] = (df["Payout ($)"] / MAX_PAYOUT * 100).round(1)
    df["Status"] = df["Heat Days"].apply(get_status)

    return df.sort_values("Payout ($)", ascending=False).reset_index(drop=True)


def portfolio_yearly_summary(heat_days_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the portfolio average payout for each year.

    Returns DataFrame with columns: year, avg_payout, avg_heat_days
    """
    village_cols = [c for c in heat_days_df.columns if c != "year"]
    records = []
    for _, row in heat_days_df.iterrows():
        year = int(row["year"])
        payouts = [compute_payout(row[v]) for v in village_cols]
        heat_days_vals = [int(row[v]) for v in village_cols]
        records.append(
            {
                "year": year,
                "avg_payout": np.mean(payouts),
                "avg_heat_days": np.mean(heat_days_vals),
                "max_payout": max(payouts),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
locations_df = load_locations()
thresholds_df = load_thresholds()
heat_days_df = load_heat_days()

# Build year list
available_years = sorted(heat_days_df["year"].unique().tolist(), reverse=True) if not heat_days_df.empty else []
year_options = ["Live 2026"] + [str(y) for y in available_years]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo
    logo_b64 = load_logo_base64()
    if logo_b64:
        st.markdown(
            f'<div style="text-align:center; padding: 10px 0 6px 0;">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="max-width:220px; width:100%;" alt="One Acre Shield" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<h2 style="text-align:center; color:{BRIGHT_GREEN}; '
            f'letter-spacing:2px; margin-bottom:0;">ONE ACRE SHIELD</h2>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<p style="text-align:center; color:{TEXT_SECONDARY}; font-size:0.85rem; '
        f'margin-top:2px;">Extreme Heat Household Protection</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Year selector
    selected_year_str = st.selectbox(
        "Select Season Year",
        options=year_options,
        index=0,
    )

    is_live = selected_year_str == "Live 2026"
    selected_year = None if is_live else int(selected_year_str)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Product parameters
    st.markdown(
        f"""
        <div class="product-info">
            <div style="color:{BRIGHT_GREEN}; font-weight:700; margin-bottom:8px;
                        font-size:0.9rem;">Product Parameters</div>
            <div style="margin-bottom:6px;">
                <span class="param-label">Payout per heat day:</span><br>
                <span class="param-value">${PAYOUT_PER_DAY}</span>
            </div>
            <div style="margin-bottom:6px;">
                <span class="param-label">Max payout (cap):</span><br>
                <span class="param-value">${MAX_PAYOUT} ({CAP_DAYS} days)</span>
            </div>
            <div style="margin-bottom:6px;">
                <span class="param-label">Coverage window:</span><br>
                <span class="param-value">Mar 1 – May 31 ({COVERAGE_DAYS} days)</span>
            </div>
            <div>
                <span class="param-label">Premium:</span><br>
                <span class="param-value">~${PREMIUM:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Village search filter
    village_filter = st.text_input(
        "Search Villages",
        placeholder="Type village name...",
    )

    # Footer
    st.markdown(
        f'<div style="position:fixed; bottom:12px; color:{TEXT_SECONDARY}; '
        f'font-size:0.72rem;">Niger State, Nigeria &middot; 173 villages</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Prepare data for the selected year
# ---------------------------------------------------------------------------
if is_live:
    live_df = load_live_data()
    if live_df is not None:
        # Expect live_data.csv to have at least a row with village columns
        village_cols = [c for c in live_df.columns if c != "year"]
        if len(live_df) > 0:
            heat_series = live_df.iloc[0][village_cols]
        else:
            heat_series = pd.Series(0, index=village_cols)

        # Determine season status from season_day column
        season_day = int(live_df.iloc[0].get("season_day", 0)) if len(live_df) > 0 else 0
        if season_day == 0:
            season_status = "Not Started"
            progress_text = "Starts Mar 1"
        elif season_day < COVERAGE_DAYS:
            season_status = f"Day {season_day} of {COVERAGE_DAYS}"
            progress_text = "In Progress"
        else:
            season_status = "Completed"
            progress_text = f"{COVERAGE_DAYS}/{COVERAGE_DAYS} days"
    else:
        heat_series = None
        season_status = "Not Started"
        progress_text = "Starts Mar 1"
else:
    # Historical year
    year_row = heat_days_df[heat_days_df["year"] == selected_year]
    village_cols = [c for c in heat_days_df.columns if c != "year"]
    if not year_row.empty:
        heat_series = year_row.iloc[0][village_cols]
    else:
        heat_series = pd.Series(0, index=village_cols)
    season_status = "Completed"
    progress_text = f"{COVERAGE_DAYS}/{COVERAGE_DAYS} days"


# Build village table if data is available
if heat_series is not None:
    village_table = build_village_table(heat_series, thresholds_df)
    total_payout = village_table["Payout ($)"].sum()
    worst_idx = village_table["Payout ($)"].idxmax()
    worst_village = village_table.loc[worst_idx, "Village"]
    worst_payout = village_table.loc[worst_idx, "Payout ($)"]
    worst_heat_days = village_table.loc[worst_idx, "Heat Days"]
else:
    village_table = None
    total_payout = 0
    worst_village = "—"
    worst_payout = 0
    worst_heat_days = 0


# ---------------------------------------------------------------------------
# Main content area — Title
# ---------------------------------------------------------------------------
title_year = "Live 2026" if is_live else str(selected_year)
st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY}; font-size:1.6rem; margin-bottom:2px;">'
    f'Extreme Heat Dashboard — <span style="color:{BRIGHT_GREEN};">{title_year}</span>'
    f'</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{TEXT_SECONDARY}; font-size:0.88rem; margin-top:0;">'
    f'Parametric heat insurance &middot; Niger State, Nigeria &middot; 173 villages'
    f'</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Live 2026 info banner
# ---------------------------------------------------------------------------
if is_live and heat_series is None:
    st.markdown(
        f"""
        <div class="info-banner">
            <strong style="color:{BRIGHT_GREEN};">Season starts March 1, 2026</strong><br>
            <span style="color:{TEXT_SECONDARY};">
                Live data will update automatically via Google Earth Engine
                once the coverage window begins.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Row 1: KPI Metric Cards
# ---------------------------------------------------------------------------
def render_kpi_card(label: str, value: str, sub: str, value_color: str = BRIGHT_GREEN):
    """Render a single KPI card as HTML."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{value_color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


# Determine status color for season status badge
if season_status == "Completed":
    season_status_color = BRIGHT_GREEN
elif season_status == "Not Started":
    season_status_color = TEXT_SECONDARY
else:
    # "Day X of 92" — in progress
    season_status_color = WARNING

# Determine worst village payout color
worst_status = get_status(worst_heat_days)
worst_color = get_status_color(worst_status) if worst_payout > 0 else BRIGHT_GREEN

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        render_kpi_card(
            "Season Status",
            season_status,
            f"Mar 1 – May 31, {title_year}",
            season_status_color,
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    payout_color = BRIGHT_GREEN if total_payout == 0 else (WARNING if total_payout < 1000 else DANGER)
    st.markdown(
        render_kpi_card(
            "Total Portfolio Payout",
            f"${total_payout:,.0f}",
            f"173 villages",
            payout_color,
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        render_kpi_card(
            "Worst Village",
            f"${worst_payout:.0f}",
            worst_village,
            worst_color,
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        render_kpi_card(
            "Season Progress",
            progress_text,
            f"Coverage: {COVERAGE_DAYS} days",
            BRIGHT_GREEN,
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Row 2: Historical Chart + Map
# ---------------------------------------------------------------------------
chart_col, map_col = st.columns([3, 2])

# --- Left: Plotly bar chart of portfolio avg payout by year ---
with chart_col:
    st.markdown(
        '<div class="section-header">Portfolio Average Payout by Year</div>',
        unsafe_allow_html=True,
    )

    if not heat_days_df.empty:
        yearly = portfolio_yearly_summary(heat_days_df)

        # Color each bar by severity
        bar_colors = []
        for _, row in yearly.iterrows():
            p = row["avg_payout"]
            if p == 0:
                bar_colors.append(SAFE_GREEN)
            elif p < 20:
                bar_colors.append(BRIGHT_GREEN)
            elif p < 40:
                bar_colors.append(WARNING)
            else:
                bar_colors.append(DANGER)

        # Highlight selected year with brighter border
        border_colors = []
        border_widths = []
        for _, row in yearly.iterrows():
            if not is_live and row["year"] == selected_year:
                border_colors.append("#ffffff")
                border_widths.append(2.5)
            else:
                border_colors.append("rgba(0,0,0,0)")
                border_widths.append(0)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=yearly["year"],
                y=yearly["avg_payout"],
                marker=dict(
                    color=bar_colors,
                    line=dict(color=border_colors, width=border_widths),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg Payout: $%{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_SECONDARY, size=11),
            xaxis=dict(
                title="Year",
                tickmode="linear",
                dtick=2,
                gridcolor=BORDER,
                linecolor=BORDER,
            ),
            yaxis=dict(
                title="Avg Payout ($)",
                gridcolor=BORDER,
                linecolor=BORDER,
                zeroline=False,
            ),
            margin=dict(l=50, r=20, t=20, b=50),
            height=400,
            showlegend=False,
            bargap=0.15,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No historical data available.")


# --- Right: Folium map ---
with map_col:
    st.markdown(
        '<div class="section-header">Village Map</div>',
        unsafe_allow_html=True,
    )

    try:
        import folium
        from streamlit_folium import st_folium

        # Centre of Niger State
        center_lat = locations_df["latitude"].mean() if not locations_df.empty else 9.6
        center_lon = locations_df["longitude"].mean() if not locations_df.empty else 6.5

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles="CartoDB dark_matter",
        )

        # Add village markers
        if village_table is not None and not locations_df.empty:
            loc_dict = {
                row["name"]: (row["latitude"], row["longitude"])
                for _, row in locations_df.iterrows()
            }

            for _, vrow in village_table.iterrows():
                vname = vrow["Village"]
                coords = loc_dict.get(vname)
                if coords is None:
                    continue

                status = vrow["Status"]
                color = get_status_color(status)
                heat_d = vrow["Heat Days"]
                payout = vrow["Payout ($)"]
                threshold = vrow["Threshold (°C)"]

                popup_html = (
                    f"<div style='font-family:sans-serif; min-width:160px;'>"
                    f"<b style='font-size:13px;'>{vname}</b><br>"
                    f"<span style='color:#888;'>Threshold:</span> {threshold}°C<br>"
                    f"<span style='color:#888;'>Heat Days:</span> {heat_d}<br>"
                    f"<span style='color:#888;'>Payout:</span> ${payout}<br>"
                    f"<span style='color:{color}; font-weight:600;'>{status}</span>"
                    f"</div>"
                )

                # Scale radius by payout
                radius = 4 + (payout / MAX_PAYOUT) * 8

                folium.CircleMarker(
                    location=coords,
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1.5,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"{vname}: ${payout}",
                ).add_to(m)

        st_folium(m, width=None, height=400, returned_objects=[])

    except ImportError:
        st.warning(
            "**Map unavailable.** Install `folium` and `streamlit-folium`:\n\n"
            "```\npip install folium streamlit-folium\n```"
        )
    except Exception as e:
        st.error(f"Error rendering map: {e}")


# ---------------------------------------------------------------------------
# Row 3: Village Payout Table
# ---------------------------------------------------------------------------
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Village Payout Details</div>',
    unsafe_allow_html=True,
)

if village_table is not None:
    display_df = village_table.copy()

    # Apply village search filter
    if village_filter:
        mask = display_df["Village"].str.contains(village_filter, case=False, na=False)
        display_df = display_df[mask]

    if display_df.empty:
        st.info(f'No villages matching "{village_filter}".')
    else:
        # Summary counts above the table
        n_safe = (display_df["Status"] == "Safe").sum()
        n_triggered = (display_df["Status"] == "Triggered").sum()
        n_capped = (display_df["Status"] == "Capped").sum()
        total = len(display_df)

        summary_cols = st.columns([1, 1, 1, 3])
        with summary_cols[0]:
            st.markdown(
                f'<span class="badge-safe">Safe: {n_safe}</span>',
                unsafe_allow_html=True,
            )
        with summary_cols[1]:
            st.markdown(
                f'<span class="badge-triggered">Triggered: {n_triggered}</span>',
                unsafe_allow_html=True,
            )
        with summary_cols[2]:
            st.markdown(
                f'<span class="badge-capped">Capped: {n_capped}</span>',
                unsafe_allow_html=True,
            )
        with summary_cols[3]:
            st.markdown(
                f'<span style="color:{TEXT_SECONDARY}; font-size:0.82rem;">'
                f'Showing {total} village{"s" if total != 1 else ""}</span>',
                unsafe_allow_html=True,
            )

        # Style the Status column with color
        def style_status(val):
            """Return inline style for status values."""
            color = get_status_color(val)
            return f"color: {color}; font-weight: 600;"

        # Format display
        styled = (
            display_df.style
            .format(
                {
                    "Threshold (°C)": "{:.1f}",
                    "Payout ($)": "${:.0f}",
                    "% of Cap": "{:.1f}%",
                }
            )
            .map(style_status, subset=["Status"])
            .set_properties(**{"background-color": CARD_BG, "color": TEXT_PRIMARY})
            .set_table_styles(
                [
                    {"selector": "th", "props": [
                        ("background-color", NAVY),
                        ("color", TEXT_SECONDARY),
                        ("font-size", "0.82rem"),
                        ("text-transform", "uppercase"),
                        ("letter-spacing", "0.5px"),
                    ]},
                    {"selector": "td", "props": [
                        ("border-bottom", f"1px solid {BORDER}"),
                    ]},
                ]
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=500,
            hide_index=True,
        )
else:
    if is_live:
        st.info(
            "No live data available yet. Live data will update automatically "
            "once the coverage window begins (March 1)."
        )
    else:
        st.warning("No data available for the selected year.")
