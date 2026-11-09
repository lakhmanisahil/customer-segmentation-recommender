"""
data_loader.py
--------------
Step 1 of the pipeline: load the raw transactions and clean them.

Each row of the raw data is ONE product line on ONE invoice:
    InvoiceNo | StockCode | Description | Quantity | InvoiceDate | UnitPrice | CustomerID | Country
"""
import pandas as pd

from src.config import RAW_FILE_CANDIDATES


def find_raw_file():
    """Return the first raw data file that exists, or raise a helpful error."""
    for path in RAW_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No dataset found in data/raw/.\n"
        "Run:  python scripts/download_data.py        (real UCI data)\n"
        "or:   python scripts/make_synthetic_data.py  (offline practice data)"
    )


def load_raw(path=None):
    """Read the raw file (Excel or CSV) into a DataFrame."""
    path = path or find_raw_file()
    print(f"[load] reading {path.name} ...")
    if str(path).endswith(".xlsx"):
        df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    else:
        df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str},
                         encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print(f"[load] {len(df):,} rows, {df['CustomerID'].nunique():,} customers (before cleaning)")
    return df


def clean(df):
    """
    Remove rows that would distort customer behaviour:
      1. rows with no CustomerID  -> we can't assign them to anyone
      2. cancelled invoices       -> InvoiceNo starts with 'C'
      3. zero/negative quantity or price -> returns, adjustments, errors
      4. exact duplicate rows
    Then add TotalPrice = Quantity * UnitPrice.
    """
    start = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print(f"[clean] kept {len(df):,} of {start:,} rows | "
          f"{df['CustomerID'].nunique():,} customers | "
          f"{df['StockCode'].nunique():,} products")
    return df.reset_index(drop=True)


def load_clean_data(path=None):
    """Convenience: load + clean in one call."""
    return clean(load_raw(path))
