def create_sales_features(df, train_mask):
    train_df = df[train_mask]
    stats = train_df.groupby("store_id")["sales"].agg(["mean", "sum"]).reset_index()
    stats.columns = ["store_id", "weekly_avg", "monthly_total"]
    return df.merge(stats, on="store_id", how="left")


def add_customer_features(transactions, customers, cutoff_date):
    historical = transactions[transactions["date"] < cutoff_date]
    customer_stats = (
        historical.groupby("customer_id").agg({"amount": ["mean", "sum", "count"]}).reset_index()
    )
    customer_stats.columns = ["customer_id", "avg_amount", "total_amount", "txn_count"]
    return customers.merge(customer_stats, on="customer_id", how="left")
