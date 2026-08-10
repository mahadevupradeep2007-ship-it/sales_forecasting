import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales & Profit Analysis")

st.write(
    "Interactive analysis of the Superstore dataset "
    "using Python, Pandas and Matplotlib."
)

@st.cache_data
def load_data():
    path = kagglehub.dataset_download(
        "vivek468/superstore-dataset-final"
    )

    csv_file = os.path.join(
        path,
        "Sample - Superstore.csv"
    )

    return pd.read_csv(
        csv_file,
        encoding="cp1252"
    )

def prepare_data(df):
    x = df[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Category",
            "Sub-Category",
            "Region"
        ]
    ]

    x = pd.get_dummies(
        x,
        columns=[
            "Category",
            "Sub-Category",
            "Region"
        ],
        dtype=float
    )

    y = df["Profit"]

    return x, y

with st.spinner("Loading dataset..."):
    df = load_data()

st.header("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Average Sales", f"${df['Sales'].mean():,.2f}")
col4.metric("Average Profit", f"${df['Profit'].mean():,.2f}")

st.subheader("First 5 Rows")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.subheader("📈 Summary Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

x, y = prepare_data(df)

st.subheader("🤖 Feature Information")

col1, col2 = st.columns(2)

col1.metric("Feature Matrix Rows", x.shape[0])
col2.metric("Feature Matrix Columns", x.shape[1])

st.header("📊 Sales Analysis")

sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.bar(
    sales.index,
    sales.values
)

ax1.set_title("Total Sales by Category")
ax1.set_xlabel("Category")
ax1.set_ylabel("Sales")

plt.tight_layout()

st.pyplot(fig1)

st.subheader("Sales vs Profit")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.scatter(
    df["Sales"],
    df["Profit"],
    alpha=0.5
)

ax2.set_title("Sales vs Profit")
ax2.set_xlabel("Sales")
ax2.set_ylabel("Profit")

plt.tight_layout()

st.pyplot(fig2)

st.subheader("Correlation Heatmap")

numeric_df = df[
    [
        "Sales",
        "Profit",
        "Quantity",
        "Discount"
    ]
]

corr = numeric_df.corr()

fig3, ax3 = plt.subplots(figsize=(8, 6))

heatmap = ax3.imshow(
    corr,
    cmap="coolwarm",
    interpolation="nearest"
)

fig3.colorbar(heatmap)

ax3.set_xticks(
    range(len(corr.columns))
)

ax3.set_xticklabels(
    corr.columns,
    rotation=45
)

ax3.set_yticks(
    range(len(corr.columns))
)

ax3.set_yticklabels(
    corr.columns
)

ax3.set_title("Correlation Heatmap")

plt.tight_layout()

st.pyplot(fig3)

st.header("📌 Key Statistics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Sales",
    f"${df['Sales'].mean():,.2f}"
)

col2.metric(
    "Average Profit",
    f"${df['Profit'].mean():,.2f}"
)

col3.metric(
    "Average Discount",
    f"{df['Discount'].mean():.2%}"
)

st.success(
    "Sales and profit analysis completed successfully!"
)
