"""
config.py
---------
One place for every path and setting used in the project.
Change a value here and the whole pipeline picks it up.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"      # model files + small outputs the app needs
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# The pipeline looks for these files in data/raw/ (first one found wins)
RAW_FILE_CANDIDATES = [
    RAW_DIR / "Online Retail.xlsx",          # real UCI dataset (scripts/download_data.py)
    RAW_DIR / "online_retail.csv",           # CSV copy of the same data
    RAW_DIR / "online_retail_synthetic.csv", # offline practice data (scripts/make_synthetic_data.py)
]

# ---------- Modelling settings ----------
RANDOM_STATE = 42          # makes K-Means and PCA results repeatable
N_CLUSTERS = 3             # final number of segments (chosen with Elbow + Silhouette)
K_RANGE = range(2, 11)     # k values tried when evaluating clusters
PCA_VARIANCE_TO_KEEP = 0.90  # keep enough PCA components to explain 90% of variance

# Features fed into clustering (built in src/features.py)
CLUSTER_FEATURES = [
    "Recency",        # days since last purchase      (lower = better)
    "Frequency",      # number of distinct invoices   (higher = better)
    "Monetary",       # total money spent             (higher = better)
    "AvgOrderValue",  # Monetary / Frequency
    "UniqueProducts", # how many different products bought
    "Tenure",         # days between first and last purchase
]

# ---------- Recommender settings ----------
TOP_N_RECOMMENDATIONS = 10
EVAL_SPLIT_DATE = "2011-09-01"   # train on data before, test on purchases after
