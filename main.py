import logging
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv

# Import your custom functions
from etl.extract import get_monthly_time_range, fetch_api_data, save_raw_data
from etl.transform import load_raw_data, flattening_table_mine, homogenize_order_types, time_slots
from etl.load import load_to_curated_folder

def run_extract(base_url, api_key, project_dir):
    """Runs the entire data extraction process."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Extract Step ---")
    
    time_range = get_monthly_time_range()
    receipts, items = fetch_api_data(base_url, api_key, time_range)
    
    output_dir = project_dir / "data" / "raw"
    file_tag = time_range[0][:7]  # Use YYYY-MM as the tag
    save_raw_data(receipts, items, output_dir, file_tag)
    
    logger.info("--- Finished Extract Step ---")
    return output_dir, file_tag

def run_transform(raw_data_dir, file_tag):
    """Runs the entire data transformation process."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Transform Step ---")
    
    json_files = load_raw_data(raw_data_dir, file_tag)
    flat_table = flattening_table_mine(json_files)
    
    flat_table = homogenize_order_types(flat_table)
    flat_table = time_slots(flat_table)
    
    logger.info("--- Finished Transform Step ---")
    return flat_table

def run_load(processed_df):
    """Runs the entire data loading process."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Load Step ---")
    
    load_to_curated_folder(processed_df)
    
    logger.info("--- Finished Load Step ---")

def main():
    # --- CONFIGURE LOGGING ---
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "etl.log"),
            logging.StreamHandler()
        ]
    )
    
    # --- SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run the ETL pipeline.")
    parser.add_argument(
        '--step', 
        choices=['extract', 'transform', 'load', 'all'], 
        default='all',
        help='Specify which ETL step to run.'
    )
    args = parser.parse_args()

    # --- LOAD CONFIGURATION ---
    project_dir = Path(__file__).parent
    env_path = project_dir / "config" / "config.env"
    load_dotenv(env_path)
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("POS_API_KEY")

    if not all([base_url, api_key]):
        raise ValueError("BASE_URL and POS_API_KEY must be set in the environment.")

    # --- RUN STEPS BASED ON ARGUMENT ---
    if args.step == 'all':
        raw_dir, file_tag = run_extract(base_url, api_key, project_dir)
        transformed_df = run_transform(raw_dir, file_tag)
        run_load(transformed_df)
    elif args.step == 'extract':
        run_extract(base_url, api_key, project_dir)
    elif args.step == 'transform':
        # Assumes extract has been run and needs the latest month's data
        file_tag = get_monthly_time_range()[0][:7]
        raw_dir = project_dir / "data" / "raw"
        run_transform(raw_dir, file_tag)
    elif args.step == 'load':
        # Load requires data from the transform step first
        file_tag = get_monthly_time_range()[0][:7]
        raw_dir = project_dir / "data" / "raw"
        transformed_df = run_transform(raw_dir, file_tag)
        run_load(transformed_df)

if __name__ == "__main__":
    main()