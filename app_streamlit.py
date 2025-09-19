import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import io
from datetime import datetime, timedelta

# ------------------------------- 
# Power BI Dataset URL (your link)
# ------------------------------- 
DATASET_URL = "https://api.powerbi.com/beta/436c36aa-85c9-4a5d-8b1e-7ba4285bce82/datasets/f6f5a890-b67a-43dc-ad69-caa7b12ddc3e/rows?experience=power-bi&key=1ZUke%2FbJpB5FJkazYbb90Jmra42H5ofWdWVVQiCZp6yax7avI3HpEFx4fhMDFAVr35svnoIh69JzR10zh3uPXw%3D%3D"

# ------------------------- 
# Utility functions 
# ------------------------- 

def generate_dummy_data():
    """Generate dummy wind speed data for testing."""
    rng = pd.date_range("2023-01-01", periods=500, freq="H")
    wind_speed = np.random.normal(8, 2, size=len(rng))
    wind_dir = np.random.randint(0, 360, size=len(rng))
    return pd.DataFrame({"time": rng, "wspd": wind_speed, "wind_dir": wind_dir})

def forecast_wind(data, periods=168):
    """Create 7-day forecast using Prophet."""
    df_train = data.rename(columns={"time": "ds", "wspd": "y"})
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods, freq="H")
    forecast = model.predict(future)
    return forecast

def calculate_energy(wind_speed, radius=40, cp=0.4, rho=1.225):
    """Calculate turbine power output (W)."""
    area = np.pi * (radius**2)
    return 0.5 * rho * area * (wind_speed**3) * cp

def impact_metrics(energy_kwh):
    """Calculate environmental impact."""
    co2_saved = energy_kwh * 0.0008  # tons of CO2 per kWh
    cars_removed = energy_kwh / 4500  # approx kWh/year per car
    return co2_saved, cars_removed

def power_curve(speed, rated_power=2000):
    """Simple turbine power curve: cut-in=3, rated=12, cut-out=25"""
    if speed < 3:
        return 0
    elif 3 <= speed < 12:
        return (speed - 3) / (12 - 3) * rated_power
    elif 12 <= speed < 25:
        return rated_power
    else:
        return 0

def compute_energy(df, rated_kw=2000):
    df = df.copy()
    df["power_kW"] = df["wspd"].apply(lambda s: power_curve(s, rated_kw))
    df["energy_kWh"] = df["power_kW"] * 1  # hourly data → energy = power * 1h
    total_energy = df["energy_kWh"].sum()
    return total_energy, df

def push_to_powerbi(df, dataset_url):
    try:
        df_copy = df.copy()
        for c in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[c]):
                df_copy[c] = df_copy[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
        payload = df_copy.to_dict(orient="records")
        headers = {"Content-Type": "application/json"}
        resp = requests.post(dataset_url, data=json.dumps(payload), headers=headers, timeout=10)
        return resp.status_code in (200, 201)
    except Exception:
        return False

# ------------------------------- 
# Initialize session_state 
# ------------------------------- 
for key in ["selected_lat","selected_lon","forecast_df","model","combined","df_hist","energy_est","energy_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------------- 
# Streamlit UI 
# ------------------------------- 
st.set_page_config("🌬️ Wind Feasibility Dashboard", layout="wide")
st.title("🌬️ Wind Energy Feasibility Dashboard")

# ------------------------------- 
# Map selection 
# ------------------------------- 
st.subheader("📍 Select Location")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
st_data = st_folium(m, width=700, height=400)

if st_data and st_data.get("last_clicked"):
    st.session_state.selected_lat = st_data["last_clicked"]["lat"]
    st.session_state.selected_lon = st_data["last_clicked"]["lng"]
    st.success(f"Selected Location: Lat={st.session_state.selected_lat:.3f}, Lon={st.session_state.selected_lon:.3f}")

# ------------------------------- 
# Sidebar inputs 
# ------------------------------- 
st.sidebar.header("⚙️ Inputs")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 180)
compute_energy_flag = st.sidebar.checkbox("Estimate Energy Output", True)
rated_kw = st.sidebar.number_input("Rated Power of Turbine (kW)", 500, 5000, 2000)

# ------------------------------- 
# Analyze & Forecast 
# ------------------------------- 
if st.button("🚀 Analyze & Forecast"):
    if st.session_state.selected_lat is None or st.session_state.selected_lon is None:
        st.error("Please select a location on the map first!")
    else:
        # Historical data
        st.session_state.df_hist = generate_dummy_data()

        # Prophet forecast
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        df_train = st.session_state.df_hist.rename(columns={"time": "ds", "wspd": "y"})
        model.fit(df_train)
        st.session_state.model = model

        future = model.make_future_dataframe(periods=forecast_days*24, freq="H")
        forecast_df = model.predict(future)
        st.session_state.forecast_df = forecast_df

        # Combine historical + forecast
        st.session_state.combined = pd.merge(
            forecast_df[["ds","yhat","yhat_lower","yhat_upper"]].rename(
                columns={"ds":"time","yhat":"wspd_pred",
                         "yhat_lower":"wspd_pred_lower",
                         "yhat_upper":"wspd_pred_upper"}
            ),
            st.session_state.df_hist, on="time", how="outer"
        )

        # Energy estimate
        if compute_energy_flag:
            forecast_for_energy = forecast_df[["ds","yhat"]].rename(columns={"ds":"time","yhat":"wspd"})
            st.session_state.energy_est, st.session_state.energy_df = compute_energy(forecast_for_energy, rated_kw=rated_kw)

        st.success("✅ Analysis & Forecast Completed!")

        # Impact Metrics
        if compute_energy_flag:
            co2_saved, cars_removed = impact_metrics(st.session_state.energy_est)
            st.subheader("🌱 Environmental Impact")
            st.write(f"**CO₂ Saved per Year:** {co2_saved:,.1f} tons")
            st.write(f"**Equivalent Cars Removed:** {cars_removed:,.0f} cars")

        # Auto-push combined to Power BI
        if st.session_state.combined is not None:
            df_push = st.session_state.combined.copy()
            df_push["lat"] = st.session_state.selected_lat
            df_push["lon"] = st.session_state.selected_lon
            push_ok = push_to_powerbi(df_push[["time","wspd_pred","wspd_pred_lower","wspd_pred_upper","lat","lon"]], DATASET_URL)
            if push_ok:
                st.info("📡 Data successfully pushed to Power BI!")
            else:
                st.warning("⚠️ Failed to push data to Power BI. Check dataset URL or network.")

        # Excel Download
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                st.session_state.combined.to_excel(writer, index=False, sheet_name="WindData")
            st.download_button(
                label="⬇️ Download Data as Excel (for Power BI)",
                data=excel_buffer.getvalue(),
                file_name="wind_forecast_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Excel export not available (openpyxl not installed).")

# ------------------------------- 
# Summary Metrics 
# ------------------------------- 
st.subheader("📊 Wind Summary Metrics")
if st.session_state.df_hist is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Wind Speed (m/s)", f"{st.session_state.df_hist['wspd'].mean():.2f}")
    col2.metric("Max Wind Speed (m/s)", f"{st.session_state.df_hist['wspd'].max():.2f}")
    col3.metric("Min Wind Speed (m/s)", f"{st.session_state.df_hist['wspd'].min():.2f}")
    col4.metric("Selected Location", f"Lat={st.session_state.selected_lat:.3f}, Lon={st.session_state.selected_lon:.3f}")

    if compute_energy_flag and st.session_state.energy_est is not None:
        avg_speed = st.session_state.forecast_df["yhat"].mean()
        total_energy = st.session_state.energy_est
        capacity_factor = (total_energy / (rated_kw * forecast_days * 24)) * 100
        viability = "Excellent" if capacity_factor > 35 else "Moderate" if capacity_factor > 20 else "Poor"

        st.metric("⚡ Estimated Energy (forecast)", f"{total_energy:,.0f} kWh")
        st.metric("📈 Capacity Factor", f"{capacity_factor:.2f}%")
        st.metric("🌱 Site Viability", viability)

# ------------------------------- 
# Tabs Layout 
# ------------------------------- 
tabs = st.tabs([
    "📊 Historical Data",
    "📈 Forecast Chart",
    "🔎 Seasonal Components",
    "⚡ Energy Estimation",
    "🛠 Interactive Filters"
])

# ---------------- Historical Data Tab ----------------
with tabs[0]:
    st.subheader("📊 Historical Wind Speed Data")
    if st.session_state.df_hist is not None:
        st.dataframe(st.session_state.df_hist.head(50))
        fig_windrose = px.scatter_polar(
            st.session_state.df_hist, r="wspd", theta="wind_dir",
            color="wspd", color_continuous_scale=px.colors.sequential.Viridis,
            title="Wind Rose Chart"
        )
        st.plotly_chart(fig_windrose, use_container_width=True)

# ---------------- Forecast Tab ----------------
with tabs[1]:
    st.subheader("📈 Forecasted Wind Speeds")
    if st.session_state.combined is not None:
        chart_type = st.radio("Select Chart Type", ["Line", "Bar", "Area", "Combo", "Histogram", "Scatter"])
        forecast_range = st.slider("Select Forecast Range (Days)", 1, forecast_days, (1, min(30, forecast_days)))
        df_plot = st.session_state.combined.copy()

        last_hist_time = st.session_state.df_hist["time"].max()
        start_plot = last_hist_time
        end_plot = last_hist_time + pd.Timedelta(days=forecast_range[1])
        df_plot = df_plot[df_plot["time"].between(start_plot, end_plot)].reset_index(drop=True)

        if chart_type == "Histogram":
            fig_hist_forecast = px.histogram(df_plot, x="wspd_pred", nbins=20, title="Forecast Wind Speed Distribution")
            st.plotly_chart(fig_hist_forecast, use_container_width=True)
        elif chart_type == "Scatter":
            fig_scatter = px.scatter(df_plot, x="time", y="wspd_pred", color="wspd_pred",
                                     title="Forecast Scatter Plot of Wind Speed")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            fig = go.Figure()
            if chart_type == "Line":
                fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot.get("wspd", np.nan), mode="lines+markers", name="Historical"))
                fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot.get("wspd_pred", np.nan), mode="lines+markers", name="Forecast"))
            elif chart_type == "Bar":
                fig.add_trace(go.Bar(x=df_plot["time"], y=df_plot.get("wspd", np.nan), name="Historical"))
                fig.add_trace(go.Bar(x=df_plot["time"], y=df_plot.get("wspd_pred", np.nan), name="Forecast"))
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot.get("wspd_pred", np.nan), fill='tozeroy', name="Forecast Area"))
            elif chart_type == "Combo":
                fig.add_trace(go.Bar(x=df_plot["time"], y=df_plot.get("wspd", np.nan), name="Historical"))
                fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot.get("wspd_pred", np.nan), mode="lines+markers", name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Seasonal Components Tab ----------------
with tabs[2]:
    st.subheader("🔎 Seasonal Components")
    if st.session_state.model and st.session_state.forecast_df is not None:
        comp_fig = st.session_state.model.plot_components(st.session_state.forecast_df)
        st.pyplot(comp_fig)

# ---------------- Energy Tab ----------------
with tabs[3]:
    if compute_energy_flag and st.session_state.energy_est is not None:
        st.subheader("⚡ Energy Estimation")
        st.metric("Total Forecast Energy", f"{st.session_state.energy_est:,.0f} kWh")

        # Daily energy bar chart
        df_energy = st.session_state.energy_df.copy()
        df_energy["day"] = df_energy["time"].dt.date
        daily_energy = df_energy.groupby("day")["energy_kWh"].sum().reset_index()
        fig_bar = px.bar(daily_energy, x="day", y="energy_kWh", title="Daily Energy Output (kWh)")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Wind speed vs power curve line chart
        fig_curve = px.line(df_energy, x="wspd", y="power_kW", title="Wind Speed vs Power Output")
        st.plotly_chart(fig_curve, use_container_width=True)

# ---------------- Interactive Filters Tab ----------------
with tabs[4]:
    st.subheader("🛠 Interactive Filters")
    if st.session_state.combined is not None:
        month_filter = st.multiselect("Select Months to Highlight", options=list(range(1,13)), default=list(range(1,13)))
        st.session_state.month_filter = month_filter
        df_filtered = st.session_state.combined[st.session_state.combined["time"].dt.month.isin(month_filter)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_filtered["time"], y=df_filtered.get("wspd", np.nan), mode="lines+markers", name="Historical"))
        fig2.add_trace(go.Scatter(x=df_filtered["time"], y=df_filtered.get("wspd_pred", np.nan), mode="lines+markers", name="Forecast"))
        st.plotly_chart(fig2, use_container_width=True)
