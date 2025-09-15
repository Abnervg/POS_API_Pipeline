import os
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import numpy as np



#This function loads the raw data from JSON files without any transformations. and uses a manual approach to flatten the data.
def load_raw_data(data_dir, file_tag):
    """Loads the raw JSON data for a specific month tag."""
    receipts_path = data_dir / f"receipts_{file_tag}.json"
    items_path = data_dir / f"items_{file_tag}.json"

    logging.info("Loading raw data from %s and %s", receipts_path, items_path)
    try:
        with open(receipts_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        with open(items_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise
    logging.info("✅ Loaded %d receipts and %d items", len(receipts), len(items))
    return receipts, items

#This function transforms the raw data into a flat table using a pandas approach.
def flattening_table_mine(opened_data):
    """Transforms the raw data into a flat table."""
    receipts, items = opened_data
    #Transform into a flat table
    rows = []
    for r in receipts:
        date_time = r["receipt_date"]
        date, time_z = date_time.split("T")
        time = time_z.rstrip("Z")
        payment_types = ";".join(p["type"] for p in r.get("payments", []))

        for line in r.get("line_items", []):
            mods = ";".join(f"{m['name']}({m['option']})"
                            for m in line.get("line_modifiers", [])) or None

            rows.append({
                "receipt_number": r["receipt_number"],
                "datetime":       r["receipt_date"],
                "date":           date,
                "time":           time,
                "order_type":     r.get("order"),
                "item_name":      line.get("item_name"),
                "cost":           line.get("cost"),
                "price":          line.get("price"),
                "total_money":    line.get("total_money"),
                "modifiers":      mods,
                "payment_type":   payment_types
            })

    df = pd.DataFrame(rows)
    return df

def flatten_with_pandas(receipts_data):
    """
    Transforms the raw receipts data into a flat table using pd.json_normalize.
    """
    # 1. FLATTEN THE DATA
    # 'record_path' tells pandas to create a new row for each item in 'line_items'.
    # 'meta' specifies the top-level fields to copy into each new row.
    df = pd.json_normalize(
        receipts_data,
        record_path=['line_items'],
        meta=['receipt_number', 'receipt_date', ['order'], ['payments']]
    )

    # 2. CLEAN AND TRANSFORM THE COLUMNS
    # Rename columns for clarity
    df.rename(columns={'order': 'order_type'}, inplace=True)

    # Convert datetime and extract date/time parts
    df['datetime'] = pd.to_datetime(df['receipt_date'])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time

    # Process nested lists with .apply() and a lambda function
    df['payment_type'] = df['payments'].apply(
        lambda p_list: ";".join(p['type'] for p in p_list if isinstance(p_list, list))
    )
    df['modifiers'] = df['line_modifiers'].apply(
        lambda m_list: ";".join(f"{m['name']}({m['option']})" for m in m_list if isinstance(m_list, list)) or None
    )

    # 3. SELECT AND REORDER FINAL COLUMNS
    final_columns = [
        "receipt_number", "datetime", "date", "time", "order_type",
        "item_name", "price", "total_money", "modifiers", "payment_type"
    ]
    df_final = df.reindex(columns=final_columns)

    return df_final

def homogenize_order_types(df):
    """
    Homogenizes the order types in the DataFrame.
    """
    #Renames the numbered order_types
    cleaned_data = df.copy()
    condition = cleaned_data["order_type"].str.contains("01" ,na=False)

    cleaned_data.loc[condition, "order_type"] = "Para Llevar"

    #Renames the Mesa - order_type
    condition_mesa = cleaned_data["order_type"].str.contains("-" ,na=False)

    cleaned_data.loc[condition_mesa,"order_type"] = "Mesa 2"

    #Homogenize order types
    condition = cleaned_data["order_type"].str.contains("domicilio",na=False)
    cleaned_data.loc[condition,"order_type"] = "A domicilio"

    condition = cleaned_data["order_type"].str.contains("Llevar",na=False)
    cleaned_data.loc[condition,"order_type"] = "Para llevar"

    return cleaned_data

def homogenize_order_types_optimized(df):
    """
    Homogenizes the order types using np.select for clarity and performance.
    """
    # Make a copy to avoid changing the original DataFrame
    cleaned_df = df.copy()
    
    # 1. Define all your conditions in order of priority.
    # The first condition that matches for a row is the one that gets applied.
    conditions = [
        cleaned_df["order_type"].str.contains("-", na=False),
        cleaned_df["order_type"].str.contains("domicilio", na=False),
        cleaned_df["order_type"].str.contains("01", na=False),
        cleaned_df["order_type"].str.contains("Llevar", na=False)
    ]
    
    # 2. Define the corresponding values for each condition.
    choices = [
        "Mesa",       # Value for rows containing "-"
        "A domicilio",  # Value for rows containing "domicilio"
        "Para llevar",  # Value for rows containing "01"
        "Para llevar"   # Value for rows containing "Llevar"
    ]
    
    # 3. Apply the conditions.
    # If no condition is met, it returns the original value as the default.
    cleaned_df["order_type"] = np.select(
        conditions, 
        choices, 
        default=cleaned_df["order_type"]
    )
    
    return cleaned_df

def change_time_to_utc_minus_6(df):
    """
    Converts the 'datetime' column to UTC-6 timezone.
    """
    # Convert 'datetime' to datetime type if it's not already
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Convert to UTC-6 timezone
    df['datetime_mx'] = df['datetime'].dt.tz_localize('America/Mexico_City', ambiguous='NaT').dt.tz_convert('UTC')

    return df

def time_slots(cleaned_data):
    # Lets classify timestamps by hours
    cleaned_data['datetime'] = pd.to_datetime(
        cleaned_data['date'].astype(str) + ' ' + cleaned_data['time'].astype(str), 
        errors='coerce')

    cleaned_data["shifted_time"] = cleaned_data["datetime"] + pd.to_timedelta(-6, unit = 'h')

    # cleaned_data["shifted_time"] = pd.to_datetime(cleaned_data["shifted_time"]).dt.time

    # Create a numerical column (minutes past midnight) for binning
    cleaned_data['minutes_past_midnight'] = cleaned_data['shifted_time'].apply(lambda t: t.hour * 60 + t.minute)

    #Define the time intervals (bins) in 1 hour increments for a full day
    bins = range(0, 24 * 60 + 1, 60) 

    # Create labels for each interval (e.g., "01:30-01:45")
    labels = [f"{i//60:02d}:00-{(i+60)//60:02d}:00" for i in bins[:-1]]

    # Create the new column using pd.cut
    cleaned_data['time_slot'] = pd.cut(cleaned_data['minutes_past_midnight'],
                            bins=bins,
                            labels=labels,
                            right=False,      # Intervals are [left, right), e.g., includes 1:30 but not 1:45
                            include_lowest=True)
    
    return cleaned_data

def run_transform(receipts, items):
    """
    Orchestrates the entire data transformation process on the provided data.
    
    Args:
        receipts (list): A list of receipt dictionaries.
        items (list): A list of item dictionaries.
        
    Returns:
        pd.DataFrame: The final, transformed DataFrame.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Transform Step ---")
    
    # The function receives the raw data lists directly
    json_files = (receipts, items)
    
    # Call each of your transformation steps in order
    flat_table = flattening_table_mine(json_files)
    flat_table = homogenize_order_types(flat_table)
    flat_table = time_slots(flat_table)
    
    logger.info("--- Finished Transform Step ---")
    return flat_table

