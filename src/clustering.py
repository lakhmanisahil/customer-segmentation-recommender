"""
clustering.py
-------------
Step 3: scale -> PCA -> K-Means, plus the tools to pick k.

Why each step?
  * StandardScaler: puts every feature on the same scale (mean 0, std 1).
    Without it, Monetary (thousands) would drown out Frequency (single digits).
  * PCA: rotates the correlated features (Frequency, Monetary and
    UniqueProducts move together) into a few uncorrelated components.
    Dropping the weakest components removes noise -> cleaner clusters,
    and gives us 3 axes to plot in 3D.
  * K-Means: groups customers so each one is close to its cluster centre.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (CLUSTER_FEATURES, K_RANGE, N_CLUSTERS,
                        PCA_VARIANCE_TO_KEEP, RANDOM_STATE)
from src.features import log_transform

SEGMENT_NAMES = {
    "A": "Champions",          # high frequency, high monetary, recent
    "B": "Budget-Conscious",   # moderate activity, smaller spend
    "C": "At-Risk",            # haven't bought in a long time, low activity
}


def preprocess(rfm, scaler=None, pca=None):
    """
    log-transform -> scale -> PCA.
    Pass a fitted scaler/pca to transform NEW customers the same way
    (that's what the app does). Leave them None to fit fresh ones.
    """
    X_log = log_transform(rfm[CLUSTER_FEATURES])

    if scaler is None:
        scaler = StandardScaler().fit(X_log)
    X_scaled = scaler.transform(X_log)

    if pca is None:
        # Fit a full PCA first to see how much variance each component holds
        full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
        cum = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, PCA_VARIANCE_TO_KEEP) + 1)
        n_comp = max(n_comp, 3)  # at least 3 so we can draw a 3D plot
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X_scaled)
        print(f"[pca] keeping {n_comp} of {len(CLUSTER_FEATURES)} components "
              f"-> {pca.explained_variance_ratio_.sum():.1%} variance explained")
    X_pca = pca.transform(X_scaled)
    return X_pca, scaler, pca


def evaluate_k(X, k_range=K_RANGE):
    """
    Try several k values and record:
      * WCSS (inertia): total squared distance of points to their centre.
        Always drops as k grows; look for the 'elbow' where gains flatten.
      * Silhouette score (-1..1): how well each point fits its own cluster
        vs the nearest other cluster. Higher = better separated.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(len(X), 5000),
                               random_state=RANDOM_STATE)
        rows.append({"k": k, "WCSS": km.inertia_, "Silhouette": sil})
        print(f"[k-eval] k={k:2d}  WCSS={km.inertia_:10.1f}  silhouette={sil:.3f}")
    return pd.DataFrame(rows)


def find_elbow(k_eval):
    """
    Simple 'knee' finder: the k whose point is farthest from the straight
    line joining the first and last WCSS points.
    """
    k = k_eval["k"].to_numpy(dtype=float)
    w = k_eval["WCSS"].to_numpy(dtype=float)
    # normalise both axes to 0..1 so distance is fair
    kn = (k - k.min()) / (k.max() - k.min())
    wn = (w - w.min()) / (w.max() - w.min())
    # distance from each point to the line (0,1)->(1,0): |x + y - 1| / sqrt(2)
    dist = np.abs(kn + wn - 1) / np.sqrt(2)
    return int(k[np.argmax(dist)])


def fit_kmeans(X, n_clusters=N_CLUSTERS):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE).fit(X)
    sil = silhouette_score(X, km.labels_, sample_size=min(len(X), 5000),
                           random_state=RANDOM_STATE)
    print(f"[kmeans] k={n_clusters} | silhouette={sil:.3f}")
    return km, sil


def name_clusters(rfm_with_cluster):
    """
    K-Means just outputs 0, 1, 2 - the numbers mean nothing and can change
    between runs. We rank clusters by a 'value score' built from their
    average behaviour and give them business names:
        best  -> A Champions, middle -> B Budget-Conscious, worst -> C At-Risk
    Returns {cluster_number: "A"/"B"/"C"}.
    """
    prof = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    z = (prof - prof.mean()) / prof.std(ddof=0)
    score = z["Frequency"] + z["Monetary"] - z["Recency"]   # recent + frequent + big spend
    order = score.sort_values(ascending=False).index.tolist()
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    return {int(c): letters[i] for i, c in enumerate(order)}


def segment_profile(segmented):
    """Average behaviour + size of each segment - the table you show stakeholders."""
    prof = segmented.groupby(["Segment", "SegmentName"]).agg(
        Customers=("Recency", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
        UniqueProducts=("UniqueProducts", "mean"),
        Tenure=("Tenure", "mean"),
    ).round(1).reset_index()
    prof["ShareOfCustomers"] = (prof["Customers"] / prof["Customers"].sum()).round(3)
    total_rev = segmented["Monetary"].sum()
    rev = segmented.groupby("Segment")["Monetary"].sum() / total_rev
    prof["ShareOfRevenue"] = prof["Segment"].map(rev).round(3)
    return prof
