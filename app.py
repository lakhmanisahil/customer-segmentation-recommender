"""
app.py - Streamlit dashboard for the Customer Segmentation & Recommender.

Run locally:   streamlit run app.py
It only reads files in artifacts/ (created by `python train.py`), so it can be
deployed without the raw dataset.
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src.clustering import SEGMENT_NAMES, preprocess
from src.config import ARTIFACTS_DIR, CLUSTER_FEATURES, TOP_N_RECOMMENDATIONS
from src.recommender import STRATEGY
from src import visualize as viz

st.set_page_config(page_title="Customer Segmentation & Recommender", page_icon="🛍️",
                   layout="wide")


# ---------------- Load artifacts (cached so it only happens once) ----------------
@st.cache_resource
def load_models():
    return (joblib.load(ARTIFACTS_DIR / "scaler.joblib"),
            joblib.load(ARTIFACTS_DIR / "pca.joblib"),
            joblib.load(ARTIFACTS_DIR / "kmeans.joblib"))


@st.cache_data
def load_tables():
    t = {name: pd.read_csv(ARTIFACTS_DIR / f"{name}.csv", dtype={"StockCode": str})
         for name in ["customer_segments", "segment_profile", "k_evaluation",
                      "monthly_seasonality", "seasonal_summary", "segment_bestsellers",
                      "customer_recommendations"]}
    t["monthly_seasonality"]["Month"] = pd.to_datetime(t["monthly_seasonality"]["Month"])
    t["customer_segments"] = t["customer_segments"].set_index("CustomerID")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        t["metrics"] = json.load(f)
    return t


if not (ARTIFACTS_DIR / "metrics.json").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

scaler, pca, kmeans = load_models()
T = load_tables()
M = T["metrics"]
segs, best = T["customer_segments"], T["segment_bestsellers"]

# ---------------- Sidebar ----------------
st.sidebar.title("🛍️ Segmentation")
page = st.sidebar.radio("Go to", ["Overview", "Clusters (3D)", "Seasonality",
                                  "Recommend: existing customer", "Recommend: new customer"])
st.sidebar.caption(f"{M['customers']:,} customers · {M['products']:,} products · "
                   f"{M['date_range'][0]} → {M['date_range'][1]}")

# ---------------- Pages ----------------
if page == "Overview":
    st.title("Customer Segmentation & Recommender")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{M['customers']:,}")
    c2.metric("Segments (k)", M["n_clusters"])
    c3.metric("Silhouette score", f"{M['silhouette']:.3f}")
    c4.metric("PCA variance kept", f"{M['pca_variance_explained']:.0%}")

    st.subheader("Segment profiles")
    st.dataframe(T["segment_profile"], hide_index=True, width="stretch")
    st.plotly_chart(viz.plot_segment_profiles(T["segment_profile"]), width="stretch")

    st.subheader("What to do with each segment")
    for seg, name in SEGMENT_NAMES.items():
        st.markdown(f"**{seg} · {name}** → {STRATEGY[seg]['label']}")

    ev = M.get("recommender_evaluation")
    if ev:
        st.subheader("Recommender evaluation (time-based holdout)")
        e1, e2, e3 = st.columns(3)
        e1.metric(f"Hit rate@{ev['n']} (segment)", f"{ev['segment_hit_rate']:.1%}",
                  f"{(ev['segment_hit_rate'] - ev['global_hit_rate']) * 100:+.1f} pts vs global")
        e2.metric(f"Precision@{ev['n']} (segment)", f"{ev['segment_precision']:.1%}",
                  f"{(ev['segment_precision'] - ev['global_precision']) * 100:+.1f} pts vs global")
        e3.metric("Customers evaluated", f"{ev['customers_evaluated']:,}")

elif page == "Clusters (3D)":
    st.title("Clusters in PCA space")
    st.plotly_chart(viz.plot_clusters_3d(segs), width="stretch")
    st.subheader("How k was chosen")
    st.plotly_chart(viz.plot_k_evaluation(T["k_evaluation"], M["n_clusters"]),
                    width="stretch")
    st.caption(f"Elbow suggests k={M['elbow_k']}; best silhouette at k={M['best_silhouette_k']}.")

elif page == "Seasonality":
    st.title("Seasonality")
    st.plotly_chart(viz.plot_monthly_revenue(T["monthly_seasonality"]), width="stretch")
    st.dataframe(T["seasonal_summary"], hide_index=True, width="stretch")

elif page == "Recommend: existing customer":
    st.title("Recommendations for an existing customer")
    cid = st.selectbox("Customer ID", segs.index.tolist())
    row = segs.loc[cid]
    st.markdown(f"### Segment {row['Segment']} · {row['SegmentName']}")
    a, b, c, d = st.columns(4)
    a.metric("Recency (days)", int(row["Recency"]))
    b.metric("Frequency (orders)", int(row["Frequency"]))
    c.metric("Monetary", f"{row['Monetary']:,.2f}")
    d.metric("RFM score", str(row["RFM_Score"]))
    st.caption(f"Strategy: {STRATEGY[row['Segment']]['label']} · already-bought items removed")

    recs = T["customer_recommendations"]
    recs = recs[recs["CustomerID"] == cid].merge(
        best[best["Segment"] == row["Segment"]][["StockCode", "Description", "UnitPrice"]],
        on="StockCode", how="left")
    st.dataframe(recs[["Rank", "StockCode", "Description", "UnitPrice"]], hide_index=True,
                 width="stretch")

else:  # new customer
    st.title("Which segment is a new customer in?")
    st.write("Enter a customer's behaviour - the saved scaler, PCA and K-Means place them "
             "in a segment exactly the way the training customers were.")
    cols = st.columns(3)
    defaults = segs[CLUSTER_FEATURES].median()
    vals = {}
    for i, f in enumerate(CLUSTER_FEATURES):
        vals[f] = cols[i % 3].number_input(f, min_value=0.0, value=float(round(defaults[f], 1)))
    if st.button("Predict segment", type="primary"):
        new = pd.DataFrame([vals])
        X, _, _ = preprocess(new, scaler=scaler, pca=pca)
        cluster = int(kmeans.predict(X)[0])
        seg = M["cluster_to_segment"][str(cluster)]
        st.success(f"Segment {seg} · {SEGMENT_NAMES[seg]}  →  {STRATEGY[seg]['label']}")
        st.dataframe(best[best["Segment"] == seg].head(TOP_N_RECOMMENDATIONS)
                     [["Rank", "StockCode", "Description", "UnitPrice"]],
                     hide_index=True, width="stretch")
