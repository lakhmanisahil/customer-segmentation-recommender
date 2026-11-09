"""
train.py
--------
Runs the WHOLE project end to end:

    raw transactions -> clean -> RFM features -> scale + PCA -> pick k
    -> K-Means (3 segments) -> name segments -> seasonality
    -> segment best-sellers -> per-customer recommendations
    -> offline evaluation -> save models, tables and charts

Usage:
    python train.py
    python train.py --skip-eval      # faster, skips the recommender evaluation
"""
import argparse
import json

import joblib
import pandas as pd

from src import config
from src.clustering import (SEGMENT_NAMES, evaluate_k, find_elbow, fit_kmeans,
                            name_clusters, preprocess, segment_profile)
from src.data_loader import load_clean_data
from src.features import add_rfm_scores, build_rfm
from src.recommender import (STRATEGY, build_segment_bestsellers, customer_history,
                             evaluate, recommend)
from src.seasonality import monthly_by_segment, seasonal_summary
from src import visualize as viz


def main(skip_eval=False):
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean ---------------------------------------------------------
    df = load_clean_data()

    # 2. RFM features -----------------------------------------------------------
    rfm = build_rfm(df)
    rfm = add_rfm_scores(rfm)

    # 3. Scale + PCA, then evaluate k --------------------------------------------
    X, scaler, pca = preprocess(rfm)
    k_eval = evaluate_k(X)
    elbow_k = find_elbow(k_eval)
    best_sil_k = int(k_eval.loc[k_eval["Silhouette"].idxmax(), "k"])
    print(f"[k-eval] elbow suggests k={elbow_k}, best silhouette at k={best_sil_k}, "
          f"using k={config.N_CLUSTERS}")

    # 4. Final K-Means + business names -------------------------------------------
    km, silhouette = fit_kmeans(X, config.N_CLUSTERS)
    rfm["Cluster"] = km.labels_
    cluster_map = name_clusters(rfm)
    rfm["Segment"] = rfm["Cluster"].map(cluster_map)
    rfm["SegmentName"] = rfm["Segment"].map(lambda s: SEGMENT_NAMES.get(s, f"Segment {s}"))
    for i in range(3):
        rfm[f"PC{i + 1}"] = X[:, i]

    profile = segment_profile(rfm)
    print("\n=== Segment profile ===")
    print(profile.to_string(index=False))

    # 5. Seasonality ---------------------------------------------------------------
    monthly = monthly_by_segment(df, rfm)
    seasonal = seasonal_summary(monthly)
    print("\n=== Seasonality ===")
    print(seasonal.to_string(index=False))

    # 6. Recommender ---------------------------------------------------------------
    best = build_segment_bestsellers(df, rfm, top_k=50)
    hist = customer_history(df)
    rec_rows = []
    for cid in rfm.index:
        _, recs = recommend(cid, rfm, best, hist, n=config.TOP_N_RECOMMENDATIONS)
        for rank, code in enumerate(recs["StockCode"], start=1):
            rec_rows.append((cid, rank, code))
    customer_recs = pd.DataFrame(rec_rows, columns=["CustomerID", "Rank", "StockCode"])

    print("\n=== Top 5 best-sellers per segment ===")
    for seg, g in best.groupby("Segment"):
        print(f"\n{seg} - {SEGMENT_NAMES.get(seg, seg)} ({STRATEGY.get(seg, {}).get('label', '')})")
        print(g.head(5)[["Rank", "StockCode", "Description"]].to_string(index=False))

    # 7. Offline evaluation -----------------------------------------------------------
    eval_summary = None
    if not skip_eval:
        print(f"\n[eval] time-based test, split at {config.EVAL_SPLIT_DATE} ...")
        eval_summary = evaluate(df, config.EVAL_SPLIT_DATE)
        print(json.dumps({k: v for k, v in eval_summary.items() if k != "by_segment"}, indent=2))

    # 8. Save everything ------------------------------------------------------------
    A = config.ARTIFACTS_DIR
    joblib.dump(scaler, A / "scaler.joblib")
    joblib.dump(pca, A / "pca.joblib")
    joblib.dump(km, A / "kmeans.joblib")
    rfm.reset_index().round(4).to_csv(A / "customer_segments.csv", index=False)
    profile.to_csv(A / "segment_profile.csv", index=False)
    k_eval.round(4).to_csv(A / "k_evaluation.csv", index=False)
    monthly.to_csv(A / "monthly_seasonality.csv", index=False)
    seasonal.to_csv(A / "seasonal_summary.csv", index=False)
    best.to_csv(A / "segment_bestsellers.csv", index=False)
    customer_recs.to_csv(A / "customer_recommendations.csv", index=False)

    metrics = {
        "dataset_rows_after_cleaning": int(len(df)),
        "customers": int(len(rfm)),
        "products": int(df["StockCode"].nunique()),
        "date_range": [str(df["InvoiceDate"].min().date()), str(df["InvoiceDate"].max().date())],
        "features": config.CLUSTER_FEATURES,
        "pca_components": int(pca.n_components_),
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "n_clusters": config.N_CLUSTERS,
        "elbow_k": elbow_k,
        "best_silhouette_k": best_sil_k,
        "silhouette": round(float(silhouette), 4),
        "cluster_to_segment": {str(k): v for k, v in cluster_map.items()},
        "recommender_evaluation": eval_summary,
    }
    with open(A / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Charts (cdn = load plotly.js from the internet -> small html files)
    F = config.FIGURES_DIR
    html = dict(include_plotlyjs="cdn")
    viz.plot_k_evaluation(k_eval, config.N_CLUSTERS).write_html(F / "elbow_silhouette.html", **html)
    viz.plot_clusters_3d(rfm).write_html(F / "clusters_3d.html", **html)
    viz.plot_segment_profiles(profile).write_html(F / "segment_profiles.html", **html)
    viz.plot_monthly_revenue(monthly).write_html(F / "monthly_revenue.html", **html)
    viz.save_static_figures(k_eval, config.N_CLUSTERS, rfm, profile, monthly, F)

    write_results_md(metrics, profile, seasonal, best)
    print(f"\n[done] artifacts -> {A.relative_to(config.ROOT_DIR)}/ | "
          f"charts -> {F.relative_to(config.ROOT_DIR)}/ | summary -> reports/RESULTS.md")


def _fmt(table):
    """Show share columns as percentages in the markdown tables."""
    t = table.copy()
    for c in t.columns:
        if "Share" in c:
            t[c] = t[c].map(lambda x: f"{x:.1%}")
    return t


def write_results_md(metrics, profile, seasonal, best):
    """Auto-written results page, so the numbers on GitHub always match the last run."""
    m = metrics
    lines = [
        "# Results", "",
        f"_Auto-generated by `train.py`. Data: {m['date_range'][0]} to {m['date_range'][1]}, "
        f"{m['dataset_rows_after_cleaning']:,} clean transactions, {m['customers']:,} customers, "
        f"{m['products']:,} products._", "",
        "## Model", "",
        "| Item | Value |", "|---|---|",
        f"| Features | {', '.join(m['features'])} |",
        f"| PCA | {m['pca_components']} components, {m['pca_variance_explained']:.1%} variance kept |",
        f"| Clusters (k) | {m['n_clusters']} (elbow at k={m['elbow_k']}, best silhouette at k={m['best_silhouette_k']}) |",
        f"| Silhouette score | {m['silhouette']:.3f} |", "",
        "![k evaluation](figures/elbow_silhouette.png)", "",
        "## Segments", "",
        _fmt(profile).to_markdown(index=False, floatfmt=",.1f"), "",
        "![segments](figures/clusters_pca.png)", "",
        "![profiles](figures/segment_profiles.png)", "",
        "## Seasonality", "",
        _fmt(seasonal).to_markdown(index=False, floatfmt=",.0f"), "",
        "![seasonality](figures/monthly_revenue.png)", "",
        "## Top 5 recommended products per segment", "",
    ]
    for seg, g in best.groupby("Segment"):
        lines += [f"**{seg} · {SEGMENT_NAMES.get(seg, seg)}** ({STRATEGY.get(seg, {}).get('label', '')})", "",
                  g.head(5)[["Rank", "StockCode", "Description", "UnitPrice"]].to_markdown(index=False), ""]
    ev = m.get("recommender_evaluation")
    if ev:
        lines += [
            "## Recommender evaluation (time-based holdout)", "",
            f"Segments and product lists built on purchases before **{ev['split_date']}**; "
            f"scored on what {ev['customers_evaluated']:,} returning customers bought afterwards "
            f"(only products new to each customer count).", "",
            f"| Metric (top {ev['n']}) | Segment-based | Global best-sellers | ",
            "|---|---|---|",
            f"| Hit rate | {ev['segment_hit_rate']:.1%} | {ev['global_hit_rate']:.1%} |",
            f"| Precision | {ev['segment_precision']:.1%} | {ev['global_precision']:.1%} |", "",
        ]
        if "precision_lift_pct" in ev:
            lines += [f"Relative precision lift vs baseline: **{ev['precision_lift_pct']:+.1f}%**", ""]
    (config.ROOT_DIR / "reports" / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true", help="skip recommender evaluation")
    main(parser.parse_args().skip_eval)
