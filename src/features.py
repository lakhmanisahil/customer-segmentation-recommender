"""
features.py
-----------
Step 2: RFM feature engineering.

We turn ~400k transaction rows into ONE row per customer that describes
their behaviour:

    Recency   - days since the customer's last purchase (snapshot date - last date)
    Frequency - number of distinct invoices (shopping trips)
    Monetary  - total money spent
plus three extra behaviour features that help the clusters separate:
    AvgOrderValue  - Monetary / Frequency (big baskets vs small baskets)
    UniqueProducts - how many different products they bought (variety)
    Tenure         - days between first and last purchase (how long they've been around)
"""
import numpy as np
import pandas as pd


def build_rfm(df, snapshot_date=None):
    """Aggregate cleaned transactions into a customer-level feature table."""
    # "Today" for the analysis = the day after the last transaction in the data.
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Country=("Country", lambda s: s.mode().iat[0]),
    )

    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["Tenure"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

    rfm = rfm[["Recency", "Frequency", "Monetary", "AvgOrderValue",
               "UniqueProducts", "Tenure", "Country"]]
    print(f"[rfm] built features for {len(rfm):,} customers "
          f"(snapshot date {snapshot_date.date()})")
    return rfm


def add_rfm_scores(rfm):
    """
    Classic 1-5 RFM scoring (quintiles). Not used by K-Means, but great for
    explaining segments to non-technical people: '555' = best customer.
    """
    out = rfm.copy()
    # Recency: lower is better, so labels go 5..1
    out["R_Score"] = pd.qcut(out["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_Score"] = pd.qcut(out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_Score"] = pd.qcut(out["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_Score"] = out["R_Score"].astype(str) + out["F_Score"].astype(str) + out["M_Score"].astype(str)
    return out


def log_transform(features):
    """
    Money and counts are heavily right-skewed (a few customers spend 100x the
    median). log1p squashes the long tail so K-Means (which uses distances)
    isn't dominated by a handful of extreme customers.
    """
    return np.log1p(features.clip(lower=0))
