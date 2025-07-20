import requests
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_monthly_time_range(tz_name="America/Mexico_City"):
    """
    Calculates a monthly time range in UTC ISO format.
    - If run on the 1st of the month (for cron), it returns the PREVIOUS month.
    - If run any other day (for manual runs), it returns the CURRENT month.
    """
    logger = logging.getLogger(__name__)
    local_tz = ZoneInfo(tz_name)
    now = datetime.now(local_tz)

    # --- CONDITIONAL LOGIC ---
    if now.day == 1:
        logger.info("First day of the month detected. Targeting previous month for automated run.")
        target_date = now - pd.DateOffset(months=1)
    else:
        logger.info("Manual run detected. Targeting current month.")
        target_date = now
    
    # The rest of the calculation uses the determined target_date
    start_of_target_month = pd.Timestamp(target_date).to_period('M').to_timestamp().tz_localize(local_tz)
    end_of_target_month = start_of_target_month + pd.offsets.MonthEnd(1)
    
    # Convert to UTC and format for the API
    utc_start = start_of_target_month.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace('+00:00', 'Z')
    utc_end = end_of_target_month.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace('+00:00', 'Z')
    
    return utc_start, utc_end

def fetch_api_data(base_url, api_key, time_range):
    """Fetches receipts and items from the API."""
    created_min, created_max = time_range
    headers = {"Authorization": f"Bearer {api_key}"}
    
    receipts_url = f"{base_url}/receipts?created_min={created_min}&created_max={created_max}"
    items_url = f"{base_url}/items"
    
    logging.info("Fetching data from %s to %s", created_min, created_max)
    try:
        receipts_response = requests.get(receipts_url, headers=headers)
        receipts_response.raise_for_status()
        receipts = receipts_response.json().get("receipts", [])
        
        items_response = requests.get(items_url, headers=headers)
        items_response.raise_for_status()
        items = items_response.json().get("items", [])
        
        logging.info("Fetched %d receipts and %d items.", len(receipts), len(items))
        return receipts, items
    except requests.RequestException as e:
        logging.error("API request failed: %s", e)
        raise

def save_raw_data(receipts, items, output_dir, file_tag):
    """Saves the raw data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    receipts_path = output_dir / f"receipts_{file_tag}.json"
    with receipts_path.open("w", encoding="utf-8") as f:
        json.dump(receipts, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %d receipts to %s", len(receipts), receipts_path)
    
    items_path = output_dir / f"items_{file_tag}.json"
    with items_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %d items to %s", len(items), items_path)

