
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="E-Commerce Sales Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Sales Analysis & Prediction")
st.caption("Indian Retail Sales Dataset | 2023–2025")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_retail_sales_dataset.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="coerce"
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df = df.dropna(subset=["Order Date", "Sales"])

    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Dashboard Filters")

years = sorted(df["Order Date"].dt.year.unique())

selected_years = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

categories = sorted(df["Category"].dropna().unique())

selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

regions = sorted(df["Region"].dropna().unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

filtered_df = df[
    df["Order Date"].dt.year.isin(selected_years)
    & df["Category"].isin(selected_categories)
    & df["Region"].isin(selected_regions)
]

# ---------------- KPI METRICS ----------------
total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique()
average_sales = (
    total_sales / total_orders if total_orders else 0
)
total_quantity = filtered_df["Quantity"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
col2.metric("🛒 Total Orders", f"{total_orders:,}")
col3.metric("📦 Units Sold", f"{total_quantity:,.0f}")
col4.metric("📈 Average Order Value", f"₹{average_sales:,.2f}")

st.divider()

# ---------------- MONTHLY SALES ----------------
st.subheader("📅 Monthly Sales Trend")

monthly = (
    filtered_df.groupby(
        filtered_df["Order Date"].dt.to_period("M")
    )["Sales"]
    .sum()
    .reset_index()
)

monthly["Month"] = monthly["Order Date"].dt.to_timestamp()

fig = px.line(
    monthly,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- CATEGORY ANALYSIS ----------------
st.subheader("🛍️ Sales by Category")

category_sales = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Category-wise Sales"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- REGION ANALYSIS ----------------
st.subheader("🌍 Sales by Region")

region_sales = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    title="Regional Sales Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- PAYMENT ANALYSIS ----------------
st.subheader("💳 Payment Mode Analysis")

payment_sales = (
    filtered_df.groupby("Payment Mode")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    payment_sales,
    x="Payment Mode",
    y="Sales",
    color="Payment Mode",
    title="Sales by Payment Mode"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- YEARLY COMPARISON ----------------
st.subheader("📊 Yearly Sales Comparison")

yearly_sales = (
    df.groupby(df["Order Date"].dt.year)["Sales"]
    .sum()
    .reset_index()
)

yearly_sales.columns = ["Year", "Sales"]

fig = px.bar(
    yearly_sales,
    x="Year",
    y="Sales",
    color="Year",
    title="Year-wise Sales"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SALES PREDICTION ----------------
st.subheader("🔮 Sales Prediction")

forecast_data = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)

forecast_data["Month"] = (
    forecast_data["Order Date"].dt.to_timestamp()
)

forecast_data["Month_Number"] = np.arange(
    len(forecast_data)
)

X = forecast_data[["Month_Number"]]
y = forecast_data["Sales"]

model = LinearRegression()
model.fit(X, y)

future_months = np.arange(
    len(forecast_data),
    len(forecast_data) + 3
).reshape(-1, 1)

predictions = model.predict(future_months)

future_dates = pd.date_range(
    start=forecast_data["Month"].max() + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

prediction_df = pd.DataFrame({
    "Month": future_dates.strftime("%B %Y"),
    "Predicted Sales": predictions
})

st.write("Predicted sales for the next 3 months:")

st.dataframe(
    prediction_df,
    use_container_width=True
)

# ---------------- SALES RECORDS ----------------
st.subheader("📋 Sales Records")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ---------------- DOWNLOAD ----------------
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Sales Data",
    data=csv,
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")
