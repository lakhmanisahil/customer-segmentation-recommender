"""
visualize.py
------------
Charts. Interactive Plotly versions are saved as .html (open in a browser),
static Matplotlib versions as .png (these show up in the GitHub README).
"""
import matplotlib

matplotlib.use("Agg")  # draw without a screen (works on servers / CI)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEGMENT_COLORS = {"Champions": "#2a9d8f", "Budget-Conscious": "#457b9d", "At-Risk": "#e76f51"}
SEGMENT_ORDER = ["Champions", "Budget-Conscious", "At-Risk"]


# ---------- Plotly (interactive) ----------
def plot_k_evaluation(k_eval, chosen_k):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Elbow Method (WCSS)", "Silhouette Score"))
    fig.add_trace(go.Scatter(x=k_eval["k"], y=k_eval["WCSS"], mode="lines+markers",
                             name="WCSS"), 1, 1)
    fig.add_trace(go.Scatter(x=k_eval["k"], y=k_eval["Silhouette"], mode="lines+markers",
                             name="Silhouette"), 1, 2)
    for col in (1, 2):
        fig.add_vline(x=chosen_k, line_dash="dash", line_color="gray", row=1, col=col)
    fig.update_xaxes(title_text="k (number of clusters)", dtick=1)
    fig.update_layout(title=f"Choosing k (chosen k = {chosen_k})", showlegend=False,
                      template="plotly_white", height=420)
    return fig


def plot_clusters_3d(segmented):
    fig = px.scatter_3d(
        segmented.reset_index(), x="PC1", y="PC2", z="PC3", color="SegmentName",
        color_discrete_map=SEGMENT_COLORS, category_orders={"SegmentName": SEGMENT_ORDER},
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"], opacity=0.7,
        title="Customer segments in PCA space (drag to rotate)")
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(template="plotly_white", height=650, legend_title_text="Segment")
    return fig


def plot_segment_profiles(profile):
    metrics = ["Recency", "Frequency", "Monetary"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Avg {m}" for m in metrics])
    for i, m in enumerate(metrics, start=1):
        fig.add_trace(go.Bar(
            x=profile["SegmentName"], y=profile[m],
            marker_color=[SEGMENT_COLORS.get(s, "#888") for s in profile["SegmentName"]],
            text=profile[m].round(0), textposition="outside", showlegend=False), 1, i)
    fig.update_layout(title="Segment profiles", template="plotly_white", height=420)
    return fig


def plot_monthly_revenue(monthly):
    """Left: raw revenue (who brings the money). Right: index (when each segment peaks)."""
    m = monthly[~monthly["PartialMonth"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Monthly revenue", "Seasonality index (100 = segment's average month)"))
    for name in SEGMENT_ORDER:
        s = m[m["SegmentName"] == name]
        for col, y in ((1, "Revenue"), (2, "SeasonalityIndex")):
            fig.add_trace(go.Scatter(x=s["Month"], y=s[y], mode="lines+markers", name=name,
                                     legendgroup=name, showlegend=(col == 1),
                                     line=dict(color=SEGMENT_COLORS[name])), 1, col)
    fig.add_hline(y=100, line_dash="dot", line_color="gray", row=1, col=2)
    fig.update_layout(title="Seasonality by segment", template="plotly_white", height=450)
    return fig


# ---------- Matplotlib (static PNG for README) ----------
def save_static_figures(k_eval, chosen_k, segmented, profile, monthly, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Elbow + silhouette
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(k_eval["k"], k_eval["WCSS"], "o-")
    ax[0].set(title="Elbow Method (WCSS)", xlabel="k", ylabel="WCSS")
    ax[1].plot(k_eval["k"], k_eval["Silhouette"], "o-", color="#e76f51")
    ax[1].set(title="Silhouette Score", xlabel="k", ylabel="score")
    for a in ax:
        a.axvline(chosen_k, ls="--", c="gray")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "elbow_silhouette.png", dpi=120)
    plt.close(fig)

    # 2. PCA scatter (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in SEGMENT_ORDER:
        s = segmented[segmented["SegmentName"] == name]
        ax.scatter(s["PC1"], s["PC2"], s=6, alpha=0.6, label=f"{name} ({len(s)})",
                   c=SEGMENT_COLORS[name])
    ax.set(title="Customer segments (PCA projection)", xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clusters_pca.png", dpi=120)
    plt.close(fig)

    # 3. Segment profile bars
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    colors = [SEGMENT_COLORS.get(s, "#888") for s in profile["SegmentName"]]
    for a, m in zip(ax, ["Recency", "Frequency", "Monetary"]):
        a.bar(profile["SegmentName"], profile[m], color=colors)
        a.set_title(f"Avg {m}")
        a.tick_params(axis="x", rotation=15)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "segment_profiles.png", dpi=120)
    plt.close(fig)

    # 4. Monthly revenue + seasonality index (partial last month left out)
    m = monthly[~monthly["PartialMonth"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for name in SEGMENT_ORDER:
        s = m[m["SegmentName"] == name]
        ax[0].plot(s["Month"], s["Revenue"], "o-", label=name, c=SEGMENT_COLORS[name])
        ax[1].plot(s["Month"], s["SeasonalityIndex"], "o-", label=name, c=SEGMENT_COLORS[name])
    ax[0].set(title="Monthly revenue by segment", ylabel="Revenue")
    ax[1].set(title="Seasonality index (100 = segment's average month)", ylabel="index")
    ax[1].axhline(100, ls=":", c="gray")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_revenue.png", dpi=120)
    plt.close(fig)
