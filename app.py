import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Ecommerce Dashboard", layout="wide")

st.title("🛒 AI Powered Ecommerce Sales Dashboard")

# Load Data
df = pd.read_excel("sales_data.xlsx")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Create Sales Column
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# Sidebar Filters
st.sidebar.header("🔎 Filter Data")

category_filter = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    df["Product"].unique(),
    default=df["Product"].unique()
)

filtered_df = df[
    (df["Category"].isin(category_filter)) &
    (df["Region"].isin(region_filter)) &
    (df["Product"].isin(product_filter))
]

# KPI Metrics
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()
avg_profit = filtered_df["Profit"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"₹ {total_sales:,.0f}")
col2.metric("📈 Total Profit", f"₹ {total_profit:,.0f}")
col3.metric("🛍 Total Orders", total_orders)
col4.metric("📊 Avg Profit", f"₹ {avg_profit:,.0f}")

st.markdown("---")

# Sales by Category
st.subheader("📦 Sales by Category")
category_sales = filtered_df.groupby("Category")["Sales"].sum()
st.bar_chart(category_sales)

# Sales by Region
st.subheader("🌍 Sales by Region")
region_sales = filtered_df.groupby("Region")["Sales"].sum()
st.bar_chart(region_sales)

# Top 5 Products
st.subheader("🏆 Top 5 Products by Sales")
top_products = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5)
st.bar_chart(top_products)

# Monthly Trend
st.subheader("📅 Monthly Sales Trend")
filtered_df["Month"] = filtered_df["Order_Date"].dt.to_period("M")
monthly_sales = filtered_df.groupby("Month")["Sales"].sum()
st.line_chart(monthly_sales)

st.markdown("---")

# 🤖 ML Sales Prediction Section
st.subheader("🤖 AI Future Sales Prediction")

product_selected = st.selectbox("Select Product for Prediction", df["Product"].unique())

product_df = df[df["Product"] == product_selected]

product_df = product_df.sort_values("Order_Date")
product_df["Month_Num"] = np.arange(len(product_df))

if len(product_df) > 1:
    X = product_df[["Month_Num"]]
    y = product_df["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[len(product_df)]])
    predicted_sales = model.predict(next_month)[0]

    st.success(f"📈 Predicted Next Month Sales for {product_selected}: ₹ {predicted_sales:,.0f}")

    # Plot Prediction
    fig, ax = plt.subplots()
    ax.plot(product_df["Month_Num"], y)
    ax.scatter(len(product_df), predicted_sales)
    ax.set_title("Sales Prediction Trend")
    st.pyplot(fig)

else:
    st.warning("Not enough data for prediction.")

st.markdown("---")

# Show Data
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df)

# Download Button
st.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)
