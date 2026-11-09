"""
seasonality.py
--------------
Step 4: when do customers buy? Monthly / weekday / hourly patterns,
overall and per segment.
"""
import pandas as pd


def monthly_by_segment(df, segments):
    """Revenue, orders and active customers per month for each segment."""
    d = df.merge(segments[["Segment", "SegmentName"]], left_on="CustomerID",
                 right_index=True, how="inner")
    d["Month"] = d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    out = d.groupby(["Month", "Segment", "SegmentName"]).agg(
        Revenue=("TotalPrice", "sum"),
        Orders=("InvoiceNo", "nunique"),
        ActiveCustomers=("CustomerID", "nunique"),
    ).reset_index()
    out["Revenue"] = out["Revenue"].round(2)

    # The data stops part-way through its last month (UCI data ends 9 Dec 2011).
    # Flag that month so charts don't show a fake "crash" at the end.
    last = df["InvoiceDate"].max()
    out["PartialMonth"] = (out["Month"] == last.to_period("M").to_timestamp()) & (last.day < 28)

    # Seasonality index: each month vs that segment's average month (100 = average).
    # Lets you compare the SHAPE of seasonality across segments of very different size.
    full = out[~out["PartialMonth"]]
    avg = full.groupby("Segment")["Revenue"].mean()
    out["SeasonalityIndex"] = (100 * out["Revenue"] / out["Segment"].map(avg)).round(1)
    return out


def weekday_hour(df):
    """Order counts by weekday and hour (overall) - useful for campaign timing."""
    d = df.drop_duplicates("InvoiceNo").copy()
    d["Weekday"] = d["InvoiceDate"].dt.day_name()
    d["Hour"] = d["InvoiceDate"].dt.hour
    return d.groupby(["Weekday", "Hour"]).size().rename("Orders").reset_index()


def seasonal_summary(monthly):
    """Peak month per segment and the share of revenue from Q4 (Oct-Dec)."""
    rows = []
    monthly = monthly[~monthly["PartialMonth"]]
    for (seg, name), g in monthly.groupby(["Segment", "SegmentName"]):
        peak = g.loc[g["Revenue"].idxmax()]
        q4 = g[g["Month"].dt.month.isin([10, 11, 12])]["Revenue"].sum() / g["Revenue"].sum()
        rows.append({
            "Segment": seg, "SegmentName": name,
            "PeakMonth": peak["Month"].strftime("%b %Y"),
            "PeakRevenue": round(peak["Revenue"], 2),
            "Q4RevenueShare": round(q4, 3),
        })
    return pd.DataFrame(rows)
