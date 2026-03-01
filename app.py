import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Professional Sales Dashboard", layout="wide")

# ------------------------
# Custom CSS for KPI cards
# ------------------------
st.markdown("""
<style>
.kpi-card {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv("sales_data.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Sales"] = df["Quantity"] * df["Unit_Price"]
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

st.title("📊 AI Powered Ecommerce Sales Dashboard")

# ------------------------
# TOP FILTER SECTION
# ------------------------
col1, col2, col3 = st.columns(3)

with col1:
    category = st.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

with col2:
    region = st.multiselect("Select Region", df["Region"].unique(), default=df["Region"].unique())

with col3:
    product = st.multiselect("Select Product", df["Product"].unique(), default=df["Product"].unique())

filtered_df = df[
    (df["Category"].isin(category)) &
    (df["Region"].isin(region)) &
    (df["Product"].isin(product))
]

st.markdown("---")

# ------------------------
# KPI CARDS
# ------------------------
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()
profit_ratio = (total_profit / total_sales) * 100 if total_sales != 0 else 0

k1, k2, k3, k4 = st.columns(4)

k1.markdown(f'<div class="kpi-card" style="background-color:#1f77b4">💰 Total Sales<br><h2>₹ {total_sales:,.0f}</h2></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card" style="background-color:#2ca02c">📈 Total Profit<br><h2>₹ {total_profit:,.0f}</h2></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card" style="background-color:#ff7f0e">🛍 Orders<br><h2>{total_orders}</h2></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card" style="background-color:#d62728">📊 Profit Ratio<br><h2>{profit_ratio:.2f}%</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# CHARTS ROW 1
# ------------------------
c1, c2 = st.columns(2)

with c1:
    fig = px.bar(filtered_df.groupby("Category")["Sales"].sum().reset_index(),
                 x="Category", y="Sales",
                 title="Sales by Category", color="Category")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.pie(filtered_df.groupby("Region")["Sales"].sum().reset_index(),
                 values="Sales", names="Region",
                 title="Sales by Region")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# CHARTS ROW 2
# ------------------------
c3, c4 = st.columns(2)

with c3:
    fig = px.bar(filtered_df.groupby("Product")["Profit"].sum().sort_values(ascending=False).reset_index(),
                 x="Product", y="Profit",
                 title="Most Profitable Products",
                 color="Profit")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.line(filtered_df.groupby("Month")["Sales"].sum().reset_index(),
                  x="Month", y="Sales",
                  title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ------------------------
# 🤖 ML SALES PREDICTION
# ------------------------
st.subheader("🤖 AI Future Profit Prediction")

product_selected = st.selectbox("Select Product", df["Product"].unique())

product_df = df[df["Product"] == product_selected]
product_df = product_df.sort_values("Order_Date")
product_df["Time_Index"] = np.arange(len(product_df))

if len(product_df) > 1:
    X = product_df[["Time_Index"]]
    y = product_df["Profit"]

    model = LinearRegression()
    model.fit(X, y)

    future = np.array([[len(product_df)]])
    predicted_profit = model.predict(future)[0]

    st.success(f"Predicted Next Month Profit for {product_selected}: ₹ {predicted_profit:,.0f}")

else:
    st.warning("Not enough data for prediction.")

st.markdown("---")

# ------------------------
# Data Table + Download
# ------------------------
st.subheader("Filtered Data")
st.dataframe(filtered_df)

st.download_button(
    "Download Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv",
    "text/csv"
)
