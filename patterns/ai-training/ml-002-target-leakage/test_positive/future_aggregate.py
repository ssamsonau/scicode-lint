def create_sales_features(df):
    df["weekly_avg"] = df.groupby("store_id")["sales"].transform("mean")
    df["monthly_total"] = df.groupby(["store_id", "month"])["sales"].transform("sum")
    df["rank"] = df.groupby("category")["sales"].rank(method="dense")
    return df


def add_customer_features(transactions, customers):
    customer_stats = (
        transactions.groupby("customer_id")
        .agg({"amount": ["mean", "sum", "count"], "is_fraud": "mean"})
        .reset_index()
    )
    customer_stats.columns = [
        "customer_id",
        "avg_amount",
        "total_amount",
        "txn_count",
        "fraud_rate",
    ]
    return customers.merge(customer_stats, on="customer_id")
