"""
recommender.py
--------------
Step 5: segment-based recommender.

Idea: customers in the same segment behave alike, so the products that
sell best INSIDE a segment are good suggestions for anyone in it.

Each segment gets its own ranking strategy (matches the business plan):
    A  Champions        -> rank by revenue          (premium / high-value items)
    B  Budget-Conscious -> cheaper-than-median items, ranked by units sold (value items)
    C  At-Risk          -> rank by number of distinct buyers (safe, broadly popular items
                           to win them back)

For one customer:  find their segment -> take that segment's ranked list
-> remove products they've already bought (that's the cross-sell part)
-> return the top N.
"""
import pandas as pd

from src.config import TOP_N_RECOMMENDATIONS

# Stock codes that are fees/postage/adjustments, not real products
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "C2", "CRUK", "PADS", "S", "B",
                     "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2"}

STRATEGY = {
    "A": {"label": "Premium / high-value items", "sort_by": "Revenue"},
    "B": {"label": "Value items (below median price)", "sort_by": "Quantity"},
    "C": {"label": "Popular re-engagement items", "sort_by": "Buyers"},
}


def _product_stats(df):
    return df.groupby(["StockCode", "Description"]).agg(
        Quantity=("Quantity", "sum"),
        Revenue=("TotalPrice", "sum"),
        Buyers=("CustomerID", "nunique"),
        UnitPrice=("UnitPrice", "median"),
    ).reset_index()


def _clean_products(df):
    df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]
    # one description per stock code (the most common one) avoids duplicates
    desc = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    df = df.drop(columns="Description").merge(desc, left_on="StockCode", right_index=True)
    return df


def build_segment_bestsellers(df, segments, top_k=50):
    """
    Returns a DataFrame of the top_k ranked products for every segment.
    df       : cleaned transactions
    segments : customer table with a 'Segment' column (index = CustomerID)
    """
    df = _clean_products(df)
    d = df.merge(segments[["Segment"]], left_on="CustomerID", right_index=True)
    median_price = df["UnitPrice"].median()

    tables = []
    for seg, g in d.groupby("Segment"):
        stats = _product_stats(g)
        strat = STRATEGY.get(seg, {"sort_by": "Quantity"})
        if seg == "B":
            stats = stats[stats["UnitPrice"] <= median_price]
        # StockCode as tie-breaker keeps the ranking identical on every run
        stats = stats.sort_values([strat["sort_by"], "StockCode"], ascending=[False, True]).head(top_k)
        stats.insert(0, "Rank", range(1, len(stats) + 1))
        stats.insert(0, "Segment", seg)
        tables.append(stats)
    best = pd.concat(tables, ignore_index=True)
    best["Revenue"] = best["Revenue"].round(2)
    return best


def build_global_bestsellers(df, top_k=50):
    """Baseline: one list for everybody, ranked by units sold."""
    stats = _product_stats(_clean_products(df)).sort_values(
        ["Quantity", "StockCode"], ascending=[False, True])
    return stats.head(top_k).reset_index(drop=True)


def customer_history(df):
    """{CustomerID: set of StockCodes they've bought}"""
    return df.groupby("CustomerID")["StockCode"].agg(set).to_dict()


def recommend(customer_id, segments, bestsellers, history, n=TOP_N_RECOMMENDATIONS,
              exclude_purchased=True):
    """Top-n products for one customer from their segment's ranked list."""
    if customer_id not in segments.index:
        raise KeyError(f"Customer {customer_id} not found")
    seg = segments.loc[customer_id, "Segment"]
    ranked = bestsellers[bestsellers["Segment"] == seg]
    if exclude_purchased:
        bought = history.get(customer_id, set())
        ranked = ranked[~ranked["StockCode"].isin(bought)]
    return seg, ranked.head(n)[["StockCode", "Description", "UnitPrice"]].reset_index(drop=True)


def recommend_for_segment(seg, bestsellers, n=TOP_N_RECOMMENDATIONS):
    """Top-n for a brand-new customer once we know their segment."""
    ranked = bestsellers[bestsellers["Segment"] == seg]
    return ranked.head(n)[["StockCode", "Description", "UnitPrice"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Offline evaluation: does segment-based targeting beat a one-size-fits-all list?
# ----------------------------------------------------------------------------
def evaluate(df, split_date, n=TOP_N_RECOMMENDATIONS):
    """
    Time-based test (no peeking into the future):
      1. Use only purchases BEFORE split_date to segment customers and build lists.
      2. For each customer who buys again AFTER split_date, recommend n products
         they hadn't bought before.
      3. Score against what they actually bought later.

    Metrics
      HitRate@n   : % of customers who bought at least one recommended product
      Precision@n : % of recommended products that were bought
    Compared with a baseline that shows everyone the same global best-sellers.
    """
    from src.clustering import fit_kmeans, name_clusters, preprocess
    from src.features import build_rfm

    split = pd.Timestamp(split_date)
    train, test = df[df["InvoiceDate"] < split], df[df["InvoiceDate"] >= split]

    rfm = build_rfm(train)
    X, _, _ = preprocess(rfm)
    km, _ = fit_kmeans(X)
    rfm["Cluster"] = km.labels_
    rfm["Segment"] = rfm["Cluster"].map(name_clusters(rfm))

    seg_best = build_segment_bestsellers(train, rfm, top_k=200)
    glob_best = build_global_bestsellers(train, top_k=200)
    hist = customer_history(train)
    future = _clean_products(test).groupby("CustomerID")["StockCode"].agg(set).to_dict()

    rows = []
    for cid, bought_later in future.items():
        if cid not in rfm.index:
            continue  # brand-new customer, nothing to segment on
        past = hist.get(cid, set())
        new_items = bought_later - past
        if not new_items:
            continue
        _, seg_recs = recommend(cid, rfm, seg_best, hist, n=n)
        glob_recs = glob_best[~glob_best["StockCode"].isin(past)].head(n)
        s_hits = len(set(seg_recs["StockCode"]) & new_items)
        g_hits = len(set(glob_recs["StockCode"]) & new_items)
        rows.append({"CustomerID": cid, "Segment": rfm.loc[cid, "Segment"],
                     "seg_hits": s_hits, "glob_hits": g_hits})

    res = pd.DataFrame(rows)
    rate = lambda s: round(float(s.mean()), 4)  # noqa: E731  (plain float -> clean JSON)
    summary = {
        "split_date": split_date,
        "customers_evaluated": int(len(res)),
        "n": n,
        "segment_hit_rate": rate(res["seg_hits"] > 0),
        "global_hit_rate": rate(res["glob_hits"] > 0),
        "segment_precision": round(float(res["seg_hits"].sum() / (n * len(res))), 4),
        "global_precision": round(float(res["glob_hits"].sum() / (n * len(res))), 4),
    }
    if summary["global_precision"] > 0:
        summary["precision_lift_pct"] = round(
            100 * (summary["segment_precision"] / summary["global_precision"] - 1), 1)
    summary["by_segment"] = [
        {"Segment": seg, "customers": int(len(g)),
         "segment_hit_rate": rate(g["seg_hits"] > 0),
         "global_hit_rate": rate(g["glob_hits"] > 0)}
        for seg, g in res.groupby("Segment")
    ]
    return summary
