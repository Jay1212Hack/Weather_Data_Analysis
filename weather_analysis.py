
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 🎨 Enhanced UI: Custom Background & Sidebar
st.set_page_config(page_title="Weather Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .stApp {background: url("https://img.freepik.com/free-photo/3d-sunny-landscape-with-tree-bright-green-grass_1048-10630.jpg") no-repeat center center fixed;
            background-size: cover;}
        h1, h2, h3 {color: #1E4D92;}
    </style>
    """,
    unsafe_allow_html=True
)

# 📥 Load Data
df = pd.read_csv("large_weather_data.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Month"] = df["Datetime"].dt.month_name()

# 🔄 Refresh Data
if st.sidebar.button("🔄 Refresh"):
    st.sidebar.experimental_rerun()
    st.sidebar.markdown("---")
 
     # 🎯 Sidebar for City & Month Selection
st.sidebar.header("🌍 Select Filters")
selected_city = st.sidebar.selectbox("Choose City", df["City"].unique())
selected_month = st.sidebar.selectbox("Select Month", df["Month"].unique())
 
# 📊 Filter Data
filtered_data = df[(df["City"] == selected_city) & (df["Month"] == selected_month)]

# 🎨 Dashboard Title
st.title(f"🌦 Weather Analysis for {selected_city} ({selected_month})")

# 🌡️ **Weather Overview**
st.subheader("☁️ Current Weather Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡 Temperature", f"{round(filtered_data['Temperature_C'].mean(), 1)}°C")

with col2:
    st.metric("💨 Wind Speed", f"{round(filtered_data['WindSpeed_kmh'].mean(), 1)} km/h")

with col3:
    st.metric("🏭 AQI", f"{round(filtered_data['AQI'].mean(), 0)}")
 
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 🔄 **Refresh Data Button**
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()

# 🔄 Clear filters
if st.sidebar.button("❌ Clear filter"):
    st.sidebar.experimental_rerun()
    st.sidebar.markdown("---")
 
# 📥 Download Filtered Data
st.sidebar.download_button(
    label="🔽Download file",
    data=filtered_data.to_csv(index=False),
    file_name=f"{selected_city}_{selected_month}_Weather.csv",
    mime="text/csv",
   )

# ✅ Monthly Temperature Trends**
st.subheader("📈 Monthly Temperature Trends")
monthly_avg_temp = df[df["City"] == selected_city].groupby("Month")["Temperature_C"].mean().reset_index()
fig_monthly_temp = px.bar(
    monthly_avg_temp, x="Month", y="Temperature_C",
    title="🌡️ Average Temperature by Month",
    color="Temperature_C",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_monthly_temp, key="temp_trend_chart")

# ✅ **Daily Temperature Range Chart**
st.subheader("📊 Daily Temperature Range")
daily_temp = filtered_data.groupby(filtered_data["Datetime"].dt.date).agg({"Temperature_C": ["min", "max"]}).reset_index()
daily_temp.columns = ["Date", "Min Temp (°C)", "Max Temp (°C)"]
fig_temp_range = px.bar(
    daily_temp, x="Date", y=["Min Temp (°C)", "Max Temp (°C)"],
    title="Daily Temperature Variation",
    barmode="group",
    color_discrete_map={"Min Temp (°C)": "blue", "Max Temp (°C)": "red"}
)
st.plotly_chart(fig_temp_range, use_container_width=True, key="temp_range_chart")

# ✅ Create Weather Condition Column 
if "Weather_Condition" not in df.columns:
    def classify_weather(row):
        if row["Precipitation_mm"] > 5:
            return "Rainy"
        elif row["Temperature_C"] > 30 and row["Humidity_%"] < 50:
            return "Sunny"
        elif row["WindSpeed_kmh"] > 30:
            return "Windy"
        elif row["AQI"] > 150:
            return "Poor Air Quality"
        else:
            return "Cloudy"

    df["Weather_Condition"] = df.apply(classify_weather, axis=1)

# ✅ Updating new coloumn
filtered_data = df[(df["City"] == selected_city) & (df["Month"] == selected_month)]

# ✅ Weather Condition Pie Chart (Fixed)
st.subheader("☁️ Weather Condition Distribution")
fig_pie = px.pie(
    filtered_data,
    names="Weather_Condition",
    title="Weather Breakdown",
    color_discrete_sequence=px.colors.sequential.Plasma
)
st.plotly_chart(fig_pie, use_container_width=True, key="weather_pie_chart")

# ✅ **AQI Category Distribution**
st.subheader("🏭 Air Quality Index (AQI) Distribution")
bins = [0, 50, 100, 150, 200, 300, 500]
labels = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
filtered_data["AQI_Category"] = pd.cut(filtered_data["AQI"], bins=bins, labels=labels)
fig_aqi = px.histogram(
    filtered_data, x="AQI_Category", color="AQI_Category",
    title="AQI Risk Levels",
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig_aqi, use_container_width=True, key="aqi_histogram")

# ✅ **Humidity Boxplot**
st.subheader("💦 Humidity Variation Across Cities")
fig_humidity = px.box(df, x="City", y="Humidity_%", color="City", title="City-wise Humidity Levels")
st.plotly_chart(fig_humidity, use_container_width=True, key="humidity_boxplot")

# ✅ **New Section: Humidity vs. Wind Speed Scatter Plot**
fig_scatter = px.scatter(
    filtered_data, 
    x="Humidity_%", 
    y="WindSpeed_kmh",
    color="Temperature_C", 
    size=filtered_data["Temperature_C"].abs(),  # Ensure non-negative size
    title="💦 Humidity vs. Wind Speed",
    labels={"Humidity": "Humidity (%)", "WindSpeed_kmh": "Wind Speed (km/h)"},
    template="plotly_dark"
)
st.plotly_chart(fig_scatter, key="humidity_wind_chart")

# ✅ **New Section: AI-Based Weather Forecast**
st.subheader("🔮 AI-Based Weather Forecast")
city_data = df[df["City"] == selected_city]
X = city_data["Datetime"].astype(int) // 10**9  # Convert to timestamp
y = city_data["Temperature_C"]

X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

future_dates = [(X.max() + i * 86400) for i in range(1, 8)]
future_temps = model.predict(np.array(future_dates).reshape(-1, 1))

forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(future_dates, unit="s"),
    "Predicted Temperature (°C)": future_temps
})

# 📈 Plot Forecast
fig_forecast = px.line(forecast_df, x="Date", y="Predicted Temperature (°C)", title="📈 7-Day Temperature Forecast")
st.plotly_chart(fig_forecast, key="forecast_chart")

# ✅ **New Section: AI-Powered Weather Alerts**
st.subheader("⚠️ AI-Powered Weather Alerts")

def weather_alerts(temp, wind, aqi):
    alerts = []
    if temp > 35:
        alerts.append("🔥 Extreme Heat Warning!")
    if wind > 40:
        alerts.append("💨 High Wind Alert!")
    if aqi > 150:
        alerts.append("🏭 Poor Air Quality! Limit Outdoor Activity.")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ Weather conditions are normal.")

latest_weather = city_data.iloc[-1]  # Latest recorded weather
weather_alerts(latest_weather["Temperature_C"], latest_weather["WindSpeed_kmh"], latest_weather["AQI"])

from geopy.geocoders import Nominatim
import time

# Initialize geolocator
geolocator = Nominatim(user_agent="weather_dashboard")

# Function to get latitude & longitude
def get_lat_lon(city, country):
    try:
        location = geolocator.geocode(f"{city}, {country}")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        return None, None

#Adding advanced Filters in Sidebar

st.sidebar.subheader("📌 Advanced Filters")

# AQI Filter
aqi_filter = st.sidebar.slider("Select AQI Range", min_value=int(df["AQI"].min()), max_value=int(df["AQI"].max()), value=(0, 200))
filtered_data = filtered_data[(filtered_data["AQI"] >= aqi_filter[0]) & (filtered_data["AQI"] <= aqi_filter[1])]

# Temperature Filter
temp_filter = st.sidebar.slider("Select Temperature Range (°C)", min_value=int(df["Temperature_C"].min()), max_value=int(df["Temperature_C"].max()), value=(0, 40))

st.subheader("🔮 AI-Based Custom Weather Forecast")

forecast_days = st.slider("Select Forecast Period (Days)", min_value=1, max_value=14, value=7)

future_dates = [(X.max() + i * 86400) for i in range(1, forecast_days + 1)]
future_temps = model.predict(np.array(future_dates).reshape(-1, 1))

forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(future_dates, unit="s"),
    "Predicted Temperature (°C)": future_temps
})

fig_forecast = px.line(forecast_df, x="Date", y="Predicted Temperature (°C)", title=f"📈 {forecast_days}-Day Temperature Forecast")
st.plotly_chart(fig_forecast, key="forecast_chart_dynamic")

st.subheader("🌪️ Wind Speed vs. Precipitation")

fig_wind_precip = px.scatter(
    filtered_data,
    x="WindSpeed_kmh",
    y="Precipitation_mm",
    color="Temperature_C",
    size=filtered_data["Precipitation_mm"].abs(),
    title="Wind Speed vs. Precipitation (Storm Trend)",
    labels={"WindSpeed_kmh": "Wind Speed (km/h)", "Precipitation_mm": "Precipitation (mm)"},
    template="plotly_dark"
)

st.plotly_chart(fig_wind_precip, key="wind_precip_chart")

#print overall file data

st.header("Overall Summary data")
if 'filtered_data' in locals():
    st.write(filtered_data.head())
else:
    st.error("🚨 filtered_data is not defined. Check your filtering logic!")
filtered_data.head()
