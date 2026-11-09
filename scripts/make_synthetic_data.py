"""
make_synthetic_data.py
----------------------
Creates a PRACTICE dataset with exactly the same columns as the real UCI
"Online Retail" data, so you can run the whole project offline.

    python scripts/make_synthetic_data.py

It hides three customer "types" inside the data (loyal big spenders, budget
shoppers, lapsed one-timers) plus a Q4 holiday peak, so the pipeline has real
structure to discover. Use the real dataset for the results you put on your
resume - this file is only for practice and testing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_synthetic.csv"

START, END = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-09")
N_CUSTOMERS = 4350
N_PRODUCTS = 600

ADJ = ["VINTAGE", "RED", "WHITE", "PINK", "BLUE", "RETRO", "JUMBO", "SET OF 3", "HEART",
       "REGENCY", "PARTY", "CHRISTMAS", "GLASS", "WOODEN", "PAPER", "SPOTTY", "LUNCH", "PAISLEY"]
NOUN = ["LANTERN", "CAKE STAND", "TEA CUP", "BUNTING", "LUNCH BAG", "CANDLE HOLDER", "MUG",
        "DOORMAT", "STORAGE TIN", "GIFT WRAP", "ALARM CLOCK", "PHOTO FRAME", "CUSHION COVER",
        "JAM JAR", "BAKING SET", "CHALKBOARD", "HOT WATER BOTTLE", "NAPKINS", "WALL ART"]
# month weights -> Q4 holiday peak
MONTH_W = {1: .7, 2: .65, 3: .85, 4: .75, 5: .9, 6: .85, 7: .85, 8: .85, 9: 1.2,
           10: 1.4, 11: 1.8, 12: 1.0}


def make_products():
    names = set()
    while len(names) < N_PRODUCTS:
        names.add(f"{RNG.choice(ADJ)} {RNG.choice(ADJ)} {RNG.choice(NOUN)}"
                  if RNG.random() < .4 else f"{RNG.choice(ADJ)} {RNG.choice(NOUN)}")
    price = np.round(np.clip(RNG.lognormal(1.0, 0.75, N_PRODUCTS), 0.29, 39.95), 2)
    popularity = 1 / np.arange(1, N_PRODUCTS + 1) ** 0.9          # Zipf-like
    RNG.shuffle(popularity)
    codes = [str(20000 + i) + (RNG.choice(["", "A", "B"]) if RNG.random() < .2 else "")
             for i in range(N_PRODUCTS)]
    return pd.DataFrame({"StockCode": codes, "Description": RNG.permutation(sorted(names)),
                         "UnitPrice": price, "pop": popularity})


def product_weights(products, kind):
    w = products["pop"].to_numpy().copy()
    p = products["UnitPrice"].to_numpy()
    if kind == "loyal":
        w *= (p / p.mean()) ** 0.8          # leans to pricier items
    elif kind == "budget":
        w *= (p.mean() / p) ** 1.0          # leans to cheap items
    return w / w.sum()


def main():
    products = make_products()
    days = pd.date_range(START, END, freq="D")
    days = days[days.dayofweek != 5]                       # shop closed Saturdays
    day_w = np.array([MONTH_W[d.month] for d in days])

    kinds = RNG.choice(["loyal", "budget", "lapsed"], N_CUSTOMERS, p=[.2, .45, .35])
    pw = {k: product_weights(products, k) for k in ("loyal", "budget", "lapsed")}
    countries = RNG.choice(["United Kingdom", "Germany", "France", "EIRE", "Spain",
                            "Netherlands", "Belgium"], N_CUSTOMERS,
                           p=[.9, .025, .022, .015, .013, .013, .012])

    rows, inv_no = [], 536365
    for i, kind in enumerate(kinds):
        cid = 12346 + i
        if kind == "loyal":
            n_inv = RNG.poisson(16) + 4
            lo, hi = RNG.integers(0, 90), len(days)
            items, qty_scale = 18, 9
        elif kind == "budget":
            n_inv = RNG.poisson(3) + 1
            lo = RNG.integers(0, len(days) - 60)
            hi = min(len(days), lo + RNG.integers(60, 300))
            items, qty_scale = 10, 4
        else:  # lapsed
            n_inv = RNG.integers(1, 3)
            lo = RNG.integers(0, 150)
            hi = lo + RNG.integers(5, 90)
            items, qty_scale = 9, 4

        w = day_w[lo:hi] / day_w[lo:hi].sum()
        inv_days = RNG.choice(days[lo:hi], size=n_inv, p=w)
        for d in inv_days:
            ts = pd.Timestamp(d) + pd.Timedelta(hours=int(RNG.integers(8, 20)),
                                                minutes=int(RNG.integers(0, 60)))
            k = max(1, RNG.poisson(items))
            idx = RNG.choice(N_PRODUCTS, size=k, replace=False, p=pw[kind])
            qty = RNG.geometric(1 / qty_scale, size=k)
            for j, q in zip(idx, qty):
                rows.append((str(inv_no), products.at[j, "StockCode"], products.at[j, "Description"],
                             int(q), ts, products.at[j, "UnitPrice"], float(cid), countries[i]))
            inv_no += 1

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])

    # Add the messy bits real data has, so cleaning code gets exercised
    noise = df.sample(frac=0.02, random_state=1).copy()            # cancellations
    noise["InvoiceNo"] = "C" + noise["InvoiceNo"]
    noise["Quantity"] *= -1
    guests = df.sample(frac=0.15, random_state=2).copy()            # no customer id
    guests["CustomerID"] = np.nan
    post = df.drop_duplicates("InvoiceNo").sample(frac=0.05, random_state=3).copy()
    post["StockCode"], post["Description"], post["Quantity"], post["UnitPrice"] = "POST", "POSTAGE", 1, 18.0
    df = pd.concat([df, noise, guests, post]).sort_values("InvoiceDate").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows, {df['CustomerID'].nunique():,} customers -> {OUT}")


if __name__ == "__main__":
    main()
