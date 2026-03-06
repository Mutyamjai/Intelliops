COLUMN_MAP = {
    "order_id": ["order_id", "id", "order_no"],
    "product": ["product", "product_name", "item"],
    "price": ["price", "unit_price", "cost"],
    "quantity": ["quantity", "qty", "units"],
    "event_time": ["created_at", "timestamp", "date"]
}

import pandas as pd
import re
from io import StringIO

def detect_columns(df):
    mapped = {}
    df_cols = [col.lower().strip() for col in df.columns]
   # print(df_cols)
    for canonical,possible_names in COLUMN_MAP.items():
        for i,col in enumerate(df_cols):
            if col in possible_names:
                mapped[canonical] = df.columns[i]
                break
   # print(mapped)
    return mapped

def to_canonical_df(df,source_name):
    # print(source_name," ",len(df))
    
    col_map = detect_columns(df)

    canonical_df = pd.DataFrame()

    canonical_df["order_id"] = df[col_map.get("order_id")] if "order_id" in col_map else None
    canonical_df["product"] = df[col_map.get("product")].astype(str).str.lower().str.strip()
    canonical_df["price"] = pd.to_numeric(
        df[col_map.get("price")], errors="coerce"
    )
    canonical_df["source"] = source_name
    if "quantity" in col_map:
        canonical_df["quantity"] = pd.to_numeric(
            df[col_map.get("quantity")], errors="coerce"
        ).fillna(1)
    else:
        col_map["quantity"] = 1
    canonical_df["event_time"] = pd.to_datetime(
        df[col_map.get("event_time")],format = 'mixed', errors="coerce"
    )
    canonical_df["event_time"] = canonical_df["event_time"].dt.strftime('%d-%m-%Y')
    
    canonical_df["revenue"] = canonical_df["quantity"]*canonical_df["price"]
    # print("Event time NaT count:", canonical_df["event_time"].isna().sum())
    canonical_df = canonical_df.dropna(subset=["price", "event_time"])
    # print(source_name," ",len(df))
    return canonical_df.reset_index(drop=True)


def clean_bad_lines(file_path):
    cleaned_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Fix numbers like 1,299 → 1299
            line = re.sub(r'(?<=\d),(?=\d{3}\b)', '', line)
            cleaned_lines.append(line)

    return "".join(cleaned_lines)


def load_and_convert(path, source_name):
    cleaned_csv = clean_bad_lines(path)

    df = pd.read_csv(StringIO(cleaned_csv))

    canonical_df = to_canonical_df(df, source_name)
    return canonical_df