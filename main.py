import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Import functions from the extract module
from etl.extract import get_monthly_time_range, fetch_api_data, save_raw_data

def main():
    # --- CONFIGURE LOGGING HERE ---
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "etl.log"),  # Log to a file
            logging.StreamHandler()                    # Log to the console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting ETL pipeline...")

    # --- DEFINE PATHS ---
    path_dir = Path(__file__).parent # This works for a project/etl/steps.py structure where main.py is in the root

    
    # Load Configuration
    env_path = path_dir / "config" / "config.env"
    
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    load_dotenv(env_path)
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("POS_API_KEY")

    if not all([base_url, api_key]):
        raise ValueError("BASE_URL and POS_API_KEY must be set in the environment.")

    # EXTRACT
    time_range = get_monthly_time_range()
    receipts, items = fetch_api_data(base_url, api_key, time_range)
    output_dir = path_dir / "data" / "raw"
    file_tag = time_range[0][:7] # Use YYYY-MM as the tag
    save_raw_data(receipts, items, output_dir, file_tag)
    
    logger.info("Extraction complete.")

    # TRANSFORM
    logger.info("Transforming data into usable format...")
    from etl.transform import load_raw_data, flattening_table_mine, homogenize_order_types, change_time_to_utc_minus_6, time_slots
    json_files = load_raw_data(output_dir, file_tag)
    flat_table = flattening_table_mine(json_files)
    logger.info("Transformation complete. Data is now in a flat table format.")
    #Apply data cleansing
    flat_table = homogenize_order_types(flat_table)
    #flat_table = change_time_to_utc_minus_6(flat_table)
    flat_table = time_slots(flat_table)
    #Test the transformation
    logger.info("Data cleansing complete. Data is now ready for analysis.")
    sold_by_time = flat_table.groupby("order_type")["receipt_number"].nunique()
    sold_by_time.plot(kind="bar")
    plt.show()

if __name__ == "__main__":
    main()