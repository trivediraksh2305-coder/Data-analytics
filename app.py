import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

st.title("🛒 Retail Product Analytics & Popularity Prediction")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diversified_ecommerce_dataset.csv")
    return df

df = load_data()

# ---------------------------
# DATA CLEANING
# ---------------------------
df = df.dropna()

# Convert percentage columns if needed
if "Return Rate" in df.columns:
    df["Return Rate"] = df["Return Rate"].replace('%', '', regex=True).astype(float)

# ---------------------------
# DASHBOARD SECTION
# ---------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", len(df))
col2.metric("Avg Price", round(df["Price"].mean(),2))
col3.metric("Avg Discount", round(df["Discount"].mean(),2))
col4.metric("Avg Popularity", round(df["Popularity Index"].mean(),2))

# ---------------------------
# CATEGORY ANALYSIS
# ---------------------------
st.subheader("📦 Category Wise Popularity")

category_pop = df.groupby("Category")["Popularity Index"].mean()

st.bar_chart(category_pop)

# ---------------------------
# PRICE VS POPULARITY
# ---------------------------
st.subheader("💰 Price vs Popularity")

fig, ax = plt.subplots()
ax.scatter(df["Price"], df["Popularity Index"])
ax.set_xlabel("Price")
ax.set_ylabel("Popularity Index")
st.pyplot(fig)

# ---------------------------
# ML MODEL SECTION
# ---------------------------
st.subheader("🤖 Popularity Prediction (ML Model)")

# Select numeric features
features = ["Price", "Discount", "Stock Level", "Shipping Cost", "Return Rate"]

X = df[features]
y = df["Popularity Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.write("Model Accuracy (R² Score):", round(r2,2))

# ---------------------------
# USER INPUT FOR PREDICTION
# ---------------------------
st.subheader("🔮 Predict Product Popularity")

price = st.number_input("Enter Price", min_value=0.0)
discount = st.number_input("Enter Discount", min_value=0.0)
stock = st.number_input("Enter Stock Level", min_value=0.0)
shipping = st.number_input("Enter Shipping Cost", min_value=0.0)
return_rate = st.number_input("Enter Return Rate", min_value=0.0)

if st.button("Predict Popularity"):
    input_data = np.array([[price, discount, stock, shipping, return_rate]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Popularity Index: {round(prediction[0],2)}")
