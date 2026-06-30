import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    path = kagglehub.dataset_download(
        "vivek468/superstore-dataset-final"
    )

    csv_file = os.path.join(path, "Sample - Superstore.csv")

    return pd.read_csv(csv_file, encoding="cp1252")


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
        columns=["Category", "Sub-Category", "Region"],
        dtype=float
    )

    y = df["Profit"]

    return x, y


def show_summary(df):
    print("\nFirst 5 Rows")
    print(df.head())

    print("\nDataset Shape")
    print(df.shape)

    print("\nSummary Statistics")
    print(df.describe())

    print("\nAverage Sales:", round(df["Sales"].mean(), 2))
    print("Average Profit:", round(df["Profit"].mean(), 2))
    print("Average Discount:", round(df["Discount"].mean(), 2))


def plot_sales_by_category(df):
    sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(sales.index, sales.values)

    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")

    plt.tight_layout()
    plt.show()


def plot_sales_vs_profit(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["Sales"],
        df["Profit"],
        alpha=0.5
    )

    plt.title("Sales vs Profit")
    plt.xlabel("Sales")
    plt.ylabel("Profit")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    numeric_df = df[
        ["Sales", "Profit", "Quantity", "Discount"]
    ]

    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))

    heatmap = plt.imshow(
        corr,
        cmap="coolwarm",
        interpolation="nearest"
    )

    plt.colorbar(heatmap)

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def main():
    df = load_data()

    x, y = prepare_data(df)

    show_summary(df)

    plot_sales_by_category(df)
    plot_sales_vs_profit(df)
    plot_correlation_heatmap(df)

    print("\nFeature Matrix Shape:", x.shape)
    print("Target Shape:", y.shape)


if __name__ == "__main__":
    main()