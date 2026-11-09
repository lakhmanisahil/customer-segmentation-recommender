"""
Small, fast tests on a hand-made dataset. Run with:  pytest -q
"""
import pandas as pd
import pytest

from src.clustering import name_clusters
from src.data_loader import clean
from src.features import build_rfm
from src.recommender import build_segment_bestsellers, customer_history, recommend


@pytest.fixture
def raw():
    return pd.DataFrame({
        "InvoiceNo": ["1", "1", "2", "C3", "4", "5", "6"],
        "StockCode": ["A", "B", "A", "A", "C", "B", "POST"],
        "Description": ["mug", "cup", "mug", "mug", "tin", "cup", "postage"],
        "Quantity": [2, 1, 3, -1, 5, 0, 1],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-02-01",
                                       "2011-02-02", "2011-03-01", "2011-03-02", "2011-03-03"]),
        "UnitPrice": [1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 18.0],
        "CustomerID": [1, 1, 1, 1, 2, 2, None],
        "Country": ["UK"] * 7,
    })


def test_clean_removes_bad_rows(raw):
    df = clean(raw)
    assert df["CustomerID"].notna().all()            # no guests
    assert not df["InvoiceNo"].str.startswith("C").any()  # no cancellations
    assert (df["Quantity"] > 0).all()                # no zero/negative qty
    assert "TotalPrice" in df.columns
    assert len(df) == 4


def test_rfm_values(raw):
    rfm = build_rfm(clean(raw))
    # snapshot = 2011-03-02 (last date + 1 day)
    assert rfm.loc[1, "Recency"] == 29          # last buy 2011-02-01
    assert rfm.loc[1, "Frequency"] == 2         # invoices 1 and 2
    assert rfm.loc[1, "Monetary"] == pytest.approx(2 + 2 + 3)
    assert rfm.loc[2, "Recency"] == 1


def test_name_clusters_orders_by_value():
    df = pd.DataFrame({"Cluster": [0, 1, 2], "Recency": [300, 10, 60],
                       "Frequency": [1, 30, 5], "Monetary": [50, 9000, 600]})
    mapping = name_clusters(df)
    assert mapping == {1: "A", 2: "B", 0: "C"}


def test_recommend_excludes_already_bought(raw):
    df = clean(raw)
    segs = pd.DataFrame({"Segment": ["A", "A"]}, index=pd.Index([1, 2], name="CustomerID"))
    best = build_segment_bestsellers(df, segs)
    seg, recs = recommend(1, segs, best, customer_history(df), n=5)
    assert seg == "A"
    assert set(recs["StockCode"]) == {"C"}      # customer 1 already owns A and B
