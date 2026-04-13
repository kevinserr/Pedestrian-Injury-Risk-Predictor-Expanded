import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Pedestrian Risk Predictor",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.main { background: #f8f9fc; }
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.25rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.3rem; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #0f1117; line-height: 1; }
.metric-sub   { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }
.section-header {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #94a3b8; border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9; padding: 4px; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-size: 0.85rem; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #ffffff !important; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.info-box {
    background: #eff6ff; border-left: 3px solid #3b82f6;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #1e40af; margin: 0.75rem 0;
}
.footer { font-size: 0.75rem; color: #94a3b8; text-align: center; padding-top: 2rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Color palette ───────────────────────────────────────────────────────────────
NAVY  = "#0f1117"
SLATE = "#2E4057"
TEAL  = "#048A81"
CORAL = "#E07A5F"
LIGHT = "#f8f9fc"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, color="#e2e8f0")
    if title:  ax.set_title(title, fontsize=11, fontweight="600", color=NAVY, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#64748b")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#64748b")


# ── Data functions ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/Users/Marcy_Student/Desktop/Capstone/Pedestrian-Injury-Risk-Predictor-Expanded/Data/cleaned_crash_data.csv")
        df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], errors="coerce")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def build_monthly(df):
    s = (df.set_index("CRASH DATE")
           .resample("ME")["NUMBER OF PEDESTRIANS INJURED"]
           .sum())
    return s[s > 0].rename("injuries")

@st.cache_data
def build_features(df):
    m = (df.set_index("CRASH DATE")
           .resample("ME")
           .agg(total_injuries=("NUMBER OF PEDESTRIANS INJURED","sum"),
                total_crashes=("NUMBER OF PEDESTRIANS INJURED","count")))
    m = m[m["total_injuries"] > 0].copy()
    m["lag_1"]           = m["total_injuries"].shift(1)
    m["lag_12"]          = m["total_injuries"].shift(12)
    m["rolling_mean_3m"] = m["total_injuries"].shift(1).rolling(3).mean()
    m["rolling_mean_6m"] = m["total_injuries"].shift(1).rolling(6).mean()
    m["rolling_std_3m"]  = m["total_injuries"].shift(1).rolling(3).std()
    m["yoy_pct_change"]  = m["total_injuries"].pct_change(12) * 100
    m["month_num"]       = m.index.month
    m["is_peak_season"]  = m["month_num"].isin([6,7,8,9,10]).astype(int)
    return m.dropna()

@st.cache_data
def fit_arima(series):
    p = adfuller(series.dropna())[1]
    d = 0 if p < 0.05 else 1
    split = len(series) - 12
    fitted = ARIMA(series.iloc[:split], order=(1, d, 1)).fit()
    return fitted, d, split

@st.cache_data
def run_decomp(series):
    if len(series) < 24:
        return None
    return seasonal_decompose(series, model="additive", period=12)

@st.cache_data
def risk_stats(df):
    bins   = [-1, 5, 9, 15, 19, 23]
    labels = ["Overnight (0-5)","Morning Commute (6-9)",
              "Midday (10-15)","Evening Commute (16-19)","Night (20-23)"]
    df = df.copy()
    df["hour_bin"] = pd.cut(df["hour"], bins=bins, labels=labels)
    return {
        "hour": df.groupby("hour_bin")["NUMBER OF PEDESTRIANS INJURED"].mean().to_dict(),
        "boro": df.groupby("BoroName")["NUMBER OF PEDESTRIANS INJURED"].mean().to_dict(),
        "veh":  df.groupby("veh_group")["NUMBER OF PEDESTRIANS INJURED"].mean().to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚶 NYC Risk Predictor")
    st.markdown("<div style='font-size:0.78rem;color:#64748b;margin-bottom:1.5rem;'>Vision Zero · Pedestrian Safety Analysis</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.5rem;'>Scenario Simulator</div>", unsafe_allow_html=True)
    hour_bin = st.selectbox("Time of Day", ["Overnight (0-5)","Morning Commute (6-9)","Midday (10-15)","Evening Commute (16-19)","Night (20-23)"], index=3)
    borough  = st.selectbox("Borough", ["Brooklyn","Manhattan","Queens","Bronx","Staten Island"])
    vehicle  = st.selectbox("Vehicle Type", ["sedan","suv","taxi","bus","motorcycle","truck","van","bike","other"])
    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.5rem;'>Forecast Settings</div>", unsafe_allow_html=True)
    forecast_months = st.slider("Months to forecast", 3, 18, 12)
    ci_level  = st.selectbox("Confidence interval", ["90%","80%","95%"])
    ci_alpha  = {"90%": 0.10, "80%": 0.20, "95%": 0.05}[ci_level]
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#475569;'>Data: NYC Motor Vehicle Collisions<br>Model: ARIMA + Scenario Scoring</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
if df is None:
    st.error("Could not find `../Data/cleaned_crash_data.csv`. Check the data path.")
    st.stop()

monthly     = build_monthly(df)
features    = build_features(df)
stats       = risk_stats(df)
decomp      = run_decomp(monthly)
arima, d, split = fit_arima(monthly)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 1rem 0;'>
  <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.14em;color:#94a3b8;margin-bottom:0.4rem;'>NYC Vision Zero · Pedestrian Safety</div>
  <h1 style='font-family:DM Sans;font-size:2rem;font-weight:600;color:#0f1117;margin:0;'>Pedestrian Injury Risk Predictor</h1>
  <p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>Scenario-based risk scoring · ARIMA forecasting · Seasonal decomposition</p>
</div>""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────────
total_inj  = int(df["NUMBER OF PEDESTRIANS INJURED"].sum())
total_cr   = len(df)
peak_mo    = monthly.idxmax().strftime("%B %Y")
recent_yoy = features["yoy_pct_change"].iloc[-1]
yoy_arrow  = "↑" if recent_yoy > 0 else "↓"
yoy_color  = "#dc2626" if recent_yoy > 0 else "#16a34a"

k1, k2, k3, k4 = st.columns(4)
for col, label, val, sub in [
    (k1, "Total Injuries",  f"{total_inj:,}",  "Across all records"),
    (k2, "Total Crashes",   f"{total_cr:,}",   "In cleaned dataset"),
    (k3, "Peak Month",      peak_mo,            f"{int(monthly.max()):,} injuries"),
    (k4, "Year-over-Year",  f"{yoy_arrow} {abs(recent_yoy):.1f}%", "Most recent 12-mo comparison"),
]:
    with col:
        color = yoy_color if label == "Year-over-Year" else "#0f1117"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='color:{color};font-size:{"1.4rem" if label=="Peak Month" else "2rem"};'>{val}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡  Scenario Simulator",
    "📈  Forecast",
    "🔍  Decomposition",
    "🧩  Temporal Features",
])


# ─── TAB 1: SCENARIO SIMULATOR ────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Scenario-Based Risk Scoring</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Answers: <strong>"Given these crash conditions, how elevated is pedestrian injury risk?"</strong><br>
        Composite score weighted across time of day (40%), borough (35%), and vehicle type (25%).
    </div>""", unsafe_allow_html=True)

    hr = stats["hour"].get(hour_bin, 0)
    br = stats["boro"].get(borough, 0)
    vr = stats["veh"].get(vehicle, 0)
    mh = max(stats["hour"].values()) or 1
    mb = max(stats["boro"].values()) or 1
    mv = max(stats["veh"].values()) or 1

    score = (hr/mh)*0.40 + (br/mb)*0.35 + (vr/mv)*0.25
    risk_pct = round(score * 100, 1)

    if risk_pct >= 60:
        risk_label, risk_color = "HIGH RISK",      "#dc2626"
    elif risk_pct >= 35:
        risk_label, risk_color = "MODERATE RISK",  "#d97706"
    else:
        risk_label, risk_color = "LOWER RISK",     "#16a34a"

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown(f"""<div class='metric-card' style='text-align:center;padding:2rem;'>
            <div class='metric-label'>Composite Risk Score</div>
            <div style='font-family:DM Mono;font-size:3.5rem;font-weight:500;color:{risk_color};line-height:1;'>{risk_pct}%</div>
            <div style='font-size:1rem;font-weight:600;color:{risk_color};margin-top:0.4rem;'>{risk_label}</div>
            <hr style='border-color:#e2e8f0;margin:1rem 0;'>
            <div style='font-size:0.78rem;color:#64748b;text-align:left;line-height:1.8;'>
                🕐 {hour_bin}<br>📍 {borough}<br>🚗 {vehicle.title()}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        fig.patch.set_facecolor(LIGHT)
        for ax, (title, data, selected, color) in zip(axes, [
            ("Time of Day", stats["hour"], hour_bin, CORAL),
            ("Borough",     stats["boro"], borough,  TEAL),
            ("Vehicle",     stats["veh"],  vehicle,  SLATE),
        ]):
            items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            labels = [k[:13] for k, _ in items]
            values = [v for _, v in items]
            colors = [color if k == selected else "#e2e8f0" for k, _ in items]
            ax.barh(labels, values, color=colors, height=0.6)
            style_ax(ax, title=title, xlabel="Avg injuries/crash")
            ax.invert_yaxis()
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()


# ─── TAB 2: FORECAST ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>ARIMA Injury Forecast</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='info-box'>
        Answers: <strong>"How many injuries are expected over the next {forecast_months} months?"</strong><br>
        ARIMA(1,{d},1) trained on all but the last 12 months. Shaded band = {ci_level} confidence interval.
    </div>""", unsafe_allow_html=True)

    n_steps   = forecast_months + 12
    fc_obj    = arima.get_forecast(steps=n_steps)
    fc_mean   = fc_obj.predicted_mean
    fc_ci     = fc_obj.conf_int(alpha=ci_alpha)

    train_s   = monthly.iloc[:split]
    test_s    = monthly.iloc[split:]
    test_fc   = fc_mean.iloc[:len(test_s)]
    future_fc = fc_mean.iloc[len(test_s):len(test_s)+forecast_months]
    future_ci = fc_ci.iloc[len(test_s):len(test_s)+forecast_months]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(LIGHT)
    ax.plot(train_s.index, train_s.values, color=SLATE, linewidth=1.5, label="Historical (train)")
    ax.plot(test_s.index, test_s.values, color=TEAL, linewidth=1.5, label="Actual (held-out)")
    ax.plot(test_fc.index, test_fc.values, color=CORAL, linewidth=1.5, linestyle="--", alpha=0.8, label="Model fit (test)")
    ax.plot(future_fc.index, future_fc.values, color=CORAL, linewidth=2.2, label=f"Forecast ({forecast_months}mo)")
    ax.fill_between(future_fc.index, future_ci.iloc[:,0], future_ci.iloc[:,1],
                    color=CORAL, alpha=0.15, label=f"{ci_level} CI")
    ax.axvline(test_s.index[0], color="#94a3b8", linewidth=1.0, linestyle=":", label="Train/test split")
    style_ax(ax, title="Monthly Pedestrian Injury Forecast — NYC", ylabel="Injuries / month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8, framealpha=0.8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    actual  = test_s.values
    pred    = test_fc.values[:len(test_s)]
    mae     = mean_absolute_error(actual, pred)
    rmse    = np.sqrt(mean_squared_error(actual, pred))
    mape    = (abs(actual - pred) / actual).mean() * 100

    st.markdown("<br><div class='section-header'>Model Performance on Held-Out Test Year</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, lbl, val, sub in [
        (m1, "MAE",  f"{mae:.0f}",  "injuries / month"),
        (m2, "RMSE", f"{rmse:.0f}", "injuries / month"),
        (m3, "MAPE", f"{mape:.1f}%","mean abs % error"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='font-size:1.6rem;'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Forecast table
    st.markdown("<br><div class='section-header'>12-Month Forecast Table</div>", unsafe_allow_html=True)
    avg12 = monthly.tail(12).mean()
    fc_df = pd.DataFrame({
        "Month":     future_fc.index.strftime("%Y-%m"),
        "Forecast":  future_fc.values.round(0).astype(int),
        f"Lower ({ci_level})": future_ci.iloc[:,0].values.round(0).astype(int),
        f"Upper ({ci_level})": future_ci.iloc[:,1].values.round(0).astype(int),
        "vs. 12mo Avg": [f"{'↑' if x > avg12 else '↓'} {abs(x-avg12):.0f}" for x in future_fc.values],
    }).set_index("Month")
    st.dataframe(fc_df, use_container_width=True)


# ─── TAB 3: DECOMPOSITION ─────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Seasonal Decomposition</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Additive model: <strong>Observed = Trend + Seasonality + Residual</strong><br>
        Separates long-run direction from repeating annual patterns and unexplained shocks.
    </div>""", unsafe_allow_html=True)

    if decomp is None:
        st.warning("At least 24 months of data required for decomposition.")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        fig.patch.set_facecolor(LIGHT)
        for ax, (data, lbl, col, sub) in zip(axes, [
            (monthly,         "Observed",   SLATE,    "Raw monthly injury counts"),
            (decomp.trend,    "Trend",      TEAL,     "Long-run direction (smoothed)"),
            (decomp.seasonal, "Seasonality",CORAL,    "Repeating annual pattern"),
            (decomp.resid,    "Residual",   "#6B4226","Unexplained noise / anomalies"),
        ]):
            ax.plot(data.index, data.values, color=col, linewidth=1.4)
            if lbl == "Residual":
                ax.axhline(0, color=NAVY, linewidth=0.7, linestyle="--")
            style_ax(ax, ylabel=lbl)
            ax.set_title(sub, fontsize=8.5, color="#64748b", loc="right", pad=4)
        axes[0].set_title("Additive Decomposition — NYC Pedestrian Injuries",
                          fontsize=11, fontweight="600", color=NAVY, loc="left", pad=8)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

        # Seasonal bar
        st.markdown("<br><div class='section-header'>Average Seasonal Effect by Month</div>", unsafe_allow_html=True)
        sea_avg = decomp.seasonal.groupby(decomp.seasonal.index.month).mean()
        sea_avg.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(LIGHT)
        bar_cols = [CORAL if v > 0 else SLATE for v in sea_avg.values]
        ax.bar(sea_avg.index, sea_avg.values, color=bar_cols, edgecolor="white", linewidth=0.6, width=0.7)
        ax.axhline(0, color=NAVY, linewidth=0.7)
        style_ax(ax, title="Monthly Seasonal Adjustment (+ = above-trend,  − = below-trend)", ylabel="Injuries vs. trend")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Trend summary
        trend_c = decomp.trend.dropna()
        pct = ((trend_c.iloc[-1] - trend_c.iloc[0]) / trend_c.iloc[0]) * 100
        direction = "decreased" if pct < 0 else "increased"
        dc = "#16a34a" if pct < 0 else "#dc2626"
        st.markdown(f"""<div class='info-box' style='background:#f0fdf4;border-color:#16a34a;color:#14532d;'>
            📊 <strong>Trend:</strong> NYC pedestrian injuries have
            <strong style='color:{dc};'>{direction} by {abs(pct):.1f}%</strong>
            over the observed period (trend component start → end).
        </div>""", unsafe_allow_html=True)

        # Anomaly table
        resid = decomp.resid.dropna()
        anom  = resid[abs(resid - resid.mean()) > 2 * resid.std()]
        if len(anom):
            st.markdown("<br><div class='section-header'>Anomalous Months (> 2σ)</div>", unsafe_allow_html=True)
            anom_df = pd.DataFrame({
                "Month":     anom.index.strftime("%Y-%m"),
                "Residual":  anom.values.round(0).astype(int),
                "Direction": ["⬆ SPIKE" if v > 0 else "⬇ DROP" for v in anom.values],
            }).set_index("Month")
            st.dataframe(anom_df, use_container_width=True)


# ─── TAB 4: TEMPORAL FEATURES ─────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Temporal Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Lag and rolling features capture <strong>temporal dependency</strong> — how the past predicts the present.
        These features encode momentum and volatility not captured by hour/borough/vehicle alone.
    </div>""", unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        st.markdown("**Predictive power of lag features**")
        fig, axes = plt.subplots(1, 3, figsize=(7, 3))
        fig.patch.set_facecolor(LIGHT)
        for ax, (lag_col, color) in zip(axes, [("lag_1", CORAL), ("lag_12", TEAL), ("rolling_mean_3m", SLATE)]):
            pd_data = features[["total_injuries", lag_col]].dropna()
            corr = pd_data["total_injuries"].corr(pd_data[lag_col])
            ax.scatter(pd_data[lag_col], pd_data["total_injuries"], alpha=0.45, color=color, s=12)
            style_ax(ax, title=f"{lag_col}\nr={corr:.2f}")
            ax.set_xlabel("Feature value", fontsize=7)
            if lag_col == "lag_1":
                ax.set_ylabel("Injuries", fontsize=7)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close()

    with cr:
        st.markdown("**Rolling means vs. raw signal**")
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor(LIGHT)
        ax.plot(features.index, features["total_injuries"], color="#AABFD1", linewidth=0.9, alpha=0.8, label="Monthly raw")
        ax.plot(features.index, features["rolling_mean_3m"], color=CORAL, linewidth=1.8, label="3m rolling")
        ax.plot(features.index, features["rolling_mean_6m"], color=SLATE, linewidth=1.8, label="6m rolling")
        style_ax(ax, ylabel="Injuries")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=30, ha="right", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # YoY chart
    st.markdown("<br><div class='section-header'>Year-over-Year % Change</div>", unsafe_allow_html=True)
    yoy = features["yoy_pct_change"].dropna()
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor(LIGHT)
    ax.bar(yoy.index, yoy.values, color=[CORAL if v > 0 else TEAL for v in yoy.values], width=20, alpha=0.85)
    ax.axhline(0, color=NAVY, linewidth=0.8)
    style_ax(ax, title="Month-by-Month Year-over-Year Injury Change (%)", ylabel="YoY change (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature table
    st.markdown("<br><div class='section-header'>Feature Sample — Last 12 Months</div>", unsafe_allow_html=True)
    show_cols = ["total_injuries","lag_1","lag_12","rolling_mean_3m","rolling_std_3m","yoy_pct_change","is_peak_season"]
    show_df = features[show_cols].tail(12).copy().round(1)
    show_df.index = show_df.index.strftime("%Y-%m")
    st.dataframe(show_df, use_container_width=True)

    st.markdown("""<div class='info-box'>
        <code>lag_1</code> last month's count &nbsp;|&nbsp;
        <code>lag_12</code> same month last year &nbsp;|&nbsp;
        <code>rolling_mean_3m</code> short-term momentum &nbsp;|&nbsp;
        <code>rolling_std_3m</code> recent volatility &nbsp;|&nbsp;
        <code>yoy_pct_change</code> safer or worse than last year? &nbsp;|&nbsp;
        <code>is_peak_season</code> Jun–Oct structural risk flag
    </div>""", unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""<div class='footer'>
    NYC Pedestrian Injury Risk Predictor · Built for Vision Zero stakeholders ·
    Data: NYC Open Data — Motor Vehicle Collisions
</div>""", unsafe_allow_html=True)