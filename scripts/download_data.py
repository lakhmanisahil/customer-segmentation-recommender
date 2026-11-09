"""
download_data.py
----------------
Downloads the real dataset used in this project:

    "Online Retail" - UCI Machine Learning Repository (Daqing Chen, 2015)
    https://archive.ics.uci.edu/dataset/352/online+retail
    ~541,909 transactions from a UK online gift retailer, Dec 2010 - Dec 2011.
    After cleaning: ~4,300 customers  (the "4,000+ customer records").
    License: CC BY 4.0

Usage:
    python scripts/download_data.py

If the download fails (network/firewall), download the zip manually from the
link above, unzip it, and put "Online Retail.xlsx" in data/raw/.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET = RAW_DIR / "Online Retail.xlsx"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return
    print(f"Downloading {URL} (about 23 MB) ...")
    try:
        with urllib.request.urlopen(URL, timeout=120) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print("Download it manually from https://archive.ics.uci.edu/dataset/352/online+retail")
        print(f"and save 'Online Retail.xlsx' into {RAW_DIR}")
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        TARGET.write_bytes(z.read(name))
    print(f"Saved -> {TARGET}")


if __name__ == "__main__":
    main()
