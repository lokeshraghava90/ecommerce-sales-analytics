
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="E-Commerce Sales Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Sales Analysis & Prediction")
st.caption(
    "Major Project | Data Science | SIMULATED SAMPLE DATA — "
    "Not actual marketplace sales"
)

# Generate reproducible simulated data
@st.cache_data
def create_data():
    np.random.seed(42)

    dates = pd.date_range(
        start="2025-10-01", periods=365, freq="D"
    )

    platform_names = [
        "Amazon", "Flipkart", "Shopsy", "Meesho", "Myntra"
    ]

    base_units = {
        "Amazon": 150, "Flipkart": 130, "Shopsy": 70,
        "Meesho": 90, "Myntra": 80
    }

    average_price = {
        "Amazon": 700, "Flipkart": 650, "Shopsy": 300,
        "Meesho": 350, "Myntra": 900
    }

    rows = []

    for platform in platform_names:
        for date in dates:
            units = max(
                0, int(np.random.normal(base_units[platform], 20))
            )

            revenue = round(
                units * np.random.uniform(
                    average_price[platform] * 0.8,
                    average_price[platform] * 1.2
                ), 2
            )

            rows.append({
                "Date": date,
                "Platform": platform,
                "Units_Sold": units,
                "Sales_Revenue": revenue,
                "Category": np.random.choice([
                    "Electronics", "Fashion", "Home", "Beauty"
                ])
            })

    return pd.DataFrame(rows)

df = create_data()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

platforms = sorted(df["Platform"].unique())
categories = sorted(df["Category"].unique())

selected_platforms = st.sidebar.multiselect(
    "Select platforms",
    platforms,
    default=platforms
)

selected_categories = st.sidebar.multiselect(
    "Select categories",
    categories,
    default=categories
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = df[
    df["Platform"].isin(selected_platforms)
    & df["Category"].isin(selected_categories)
    & df["Date"].dt.date.between(start_date, end_date)
]

# KPI cards
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{filtered['Sales_Revenue'].sum():,.0f}")
col2.metric("Units Sold", f"{filtered['Units_Sold'].sum():,}")
col3.metric("Platforms", filtered["Platform"].nunique())
col4.metric("Records", f"{len(filtered):,}")

if filtered.empty:
    st.warning("No data matches your filters. Change the selections.")
    st.stop()

# Charts
st.subheader("Platform Performance")

revenue_by_platform = (
    filtered.groupby("Platform", as_index=False)["Sales_Revenue"]
    .sum()
)

fig_revenue = px.bar(
    revenue_by_platform,
    x="Platform",
    y="Sales_Revenue",
    color="Platform",
    title="Simulated Revenue by Platform",
    labels={"Sales_Revenue": "Revenue (₹)"}
)
st.plotly_chart(fig_revenue, use_container_width=True)

units_by_platform = (
    filtered.groupby("Platform", as_index=False)["Units_Sold"].sum()
)

fig_units = px.bar(
    units_by_platform,
    x="Platform",
    y="Units_Sold",
    color="Platform",
    title="Simulated Units Sold by Platform"
)
st.plotly_chart(fig_units, use_container_width=True)

category_sales = (
    filtered.groupby("Category", as_index=False)["Sales_Revenue"].sum()
)

fig_category = px.pie(
    category_sales,
    names="Category",
    values="Sales_Revenue",
    title="Revenue Distribution by Category"
)
st.plotly_chart(fig_category, use_container_width=True)

# Download filtered data
st.subheader("Export Data")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered Data as CSV",
    data=csv,
    file_name="simulated_ecommerce_sales.csv",
    mime="text/csv"
)

# Prediction models
st.subheader("Machine Learning Evaluation")

ml_df = df.copy()
ml_df["Day"] = ml_df["Date"].dt.day
ml_df["Month"] = ml_df["Date"].dt.month
ml_df["DayOfWeek"] = ml_df["Date"].dt.dayofweek
ml_df["DayOfYear"] = ml_df["Date"].dt.dayofyear

ml_df = pd.get_dummies(
    ml_df, columns=["Platform", "Category"], dtype=int
)

features = [
    c for c in ml_df.columns
    if c not in ["Date", "Units_Sold", "Sales_Revenue"]
]

split_date = pd.Timestamp("2026-07-01")
train_mask = ml_df["Date"] < split_date
test_mask = ml_df["Date"] >= split_date

X_train = ml_df.loc[train_mask, features]
X_test = ml_df.loc[test_mask, features]

revenue_model = RandomForestRegressor(
    n_estimators=50, max_depth=12, random_state=42, n_jobs=-1
)
units_model = RandomForestRegressor(
    n_estimators=50, max_depth=12, random_state=42, n_jobs=-1
)

revenue_model.fit(X_train, ml_df.loc[train_mask, "Sales_Revenue"])
units_model.fit(X_train, ml_df.loc[train_mask, "Units_Sold"])

rev_actual = ml_df.loc[test_mask, "Sales_Revenue"]
rev_pred = revenue_model.predict(X_test)
units_actual = ml_df.loc[test_mask, "Units_Sold"]
units_pred = units_model.predict(X_test)

m1, m2 = st.columns(2)

m1.metric(
    "Revenue MAE",
    f"₹{mean_absolute_error(rev_actual, rev_pred):,.2f}"
)
m1.metric("Revenue R²", f"{r2_score(rev_actual, rev_pred):.3f}")

m2.metric(
    "Units MAE",
    f"{mean_absolute_error(units_actual, units_pred):.2f} units"
)
m2.metric("Units R²", f"{r2_score(units_actual, units_pred):.3f}")

st.info(
    "Model scores are measured on held-out simulated data. "
    "They do not represent verified real-world marketplace accuracy."
)

# 30-day predictions
st.subheader("30-Day Simulated Predictions")

future_dates = pd.date_range(
    df["Date"].max() + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_rows = []

for platform in platforms:
    for date in future_dates:
        future_rows.append({
            "Date": date,
            "Platform": platform,
            "Category": "Electronics"
        })

future = pd.DataFrame(future_rows)

future["Day"] = future["Date"].dt.day
future["Month"] = future["Date"].dt.month
future["DayOfWeek"] = future["Date"].dt.dayofweek
future["DayOfYear"] = future["Date"].dt.dayofyear

future_encoded = pd.get_dummies(
    future, columns=["Platform", "Category"], dtype=int
)

future_X = future_encoded.reindex(columns=features, fill_value=0)

future["Predicted_Revenue"] = np.maximum(
    0, revenue_model.predict(future_X)
).round(2)

future["Predicted_Units"] = np.maximum(
    0, units_model.predict(future_X)
).round().astype(int)

f1, f2 = st.columns(2)
f1.metric("Predicted Revenue (30 days)", f"₹{future['Predicted_Revenue'].sum():,.2f}")
f2.metric("Predicted Units (30 days)", f"{future['Predicted_Units'].sum():,}")

st.dataframe(future, use_container_width=True)

fig_future = px.line(
    future,
    x="Date",
    y="Predicted_Revenue",
    color="Platform",
    title="Predicted Daily Revenue by Platform"
)
st.plotly_chart(fig_future, use_container_width=True)

prediction_csv = future.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download 30-Day Predictions",
    data=prediction_csv,
    file_name="simulated_30_day_predictions.csv",
    mime="text/csv"
)

st.caption("Project developed for academic demonstration.")
