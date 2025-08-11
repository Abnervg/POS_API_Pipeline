import logging
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
from pendulum import datetime

# Import your custom functions at the top
from etl.extract import get_monthly_time_range, fetch_api_data, save_raw_data, save_last_extraction, read_last_timestamp
from etl.transform import load_raw_data, flattening_table_mine, homogenize_order_types, time_slots
from etl.load import load_to_curated_folder, load_to_aws_bucket, load_historical_data_from_local
from reporting.monthly_report import generate_monthly_report
from reporting.data_preparation import clean_data_for_reporting, explode_combo_items_advanced
from etl.extract import fetch_api_data
#from etl.extract import get_new_receipt_count
from reporting.cumulative_report import generate_cumulative_report

def run_extract(config):
    """Runs the entire data extraction process."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Extract Step ---")
    
    time_range = get_monthly_time_range()
    receipts, items = fetch_api_data(config['base_url'], config['api_key'], time_range)
    
    output_dir = config['project_dir'] / "data" / "raw"
    file_tag = time_range[0][:7]
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

def run_load(processed_df, config, file_tag):
    """Runs the entire data loading process."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Load Step ---")
    
    # Save to local curated folder
    curated_dir = config['project_dir'] / "data" / "curated"
    load_to_curated_folder(processed_df, curated_dir, file_tag)
    
    # Save to S3
    load_to_aws_bucket(processed_df, config['s3_bucket'], file_tag)
    
    logger.info("--- Finished Load Step ---")

def run_monthly_report(config):
    """Orchestrates the monthly report generation."""
    """
    THIS IS FOR LOADING THE DATA FROM LOCAL
     # Load the final, curated data for reporting
    curated_dir = config['project_dir'] / "data" / "curated"
    curated_file_path = curated_dir / f"curated_data_{file_tag}.csv"
    if not curated_file_path.exists():
        raise FileNotFoundError(f"Curated data file not found: {curated_file_path}. Run transform/load steps first.")
    
    final_df = pd.read_csv(curated_file_path)
    """
    generate_monthly_report(config)

def run_cumulative_report(config):
    logger = logging.getLogger(__name__)
    """Generates the cumulative report based on all historical data."""
    generate_cumulative_report(config)


def run_load_historical_data(config):
    """Load historical data from local raw JSON files and merge into S3."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Historical Data Load ---")
    local_raw_dir = config['project_dir'] / "data" / "raw"
    load_historical_data_from_local(local_raw_dir, config['s3_bucket'])
    logger.info("--- Finished Historical Data Load ---")

def run_extract_historical_data(config):
    """Extracts all historical data from the API and saves it locally."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Historical Data Extraction ---")
    
    from etl.extract import fetch_all_historical_data
    from datetime import datetime, timezone
    all_receipts, all_items = fetch_all_historical_data(config['base_url'], config['api_key'])
    
    output_dir = config['project_dir'] / "data" / "raw"
    file_tag = f"historical_to_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    save_raw_data(all_receipts, all_items, output_dir, file_tag)
    
    logger.info("--- Finished Historical Data Extraction ---")

def run_check(config):
    """Checks for new data since the last successful extraction."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Data Check ---")
    
    state_file_path = config['project_dir'] / "config" / "state.json"
    last_timestamp = read_last_timestamp(state_file_path)
    
    if not last_timestamp:
        logger.info("No last successful extraction timestamp found. Cannot check for new data.")
        return
    
    new_receipt_count = get_new_receipt_count(config['base_url'], config['api_key'], last_timestamp)
    logger.info(f"New receipt count since last extraction: {new_receipt_count}")
    
    logger.info("--- Finished Data Check ---")

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
    parser = argparse.ArgumentParser(description="Run the ETL and Reporting pipeline.")
    parser.add_argument('--step', choices=['extract', 'transform', 'load', 'monthly_report', 'cumulative_report', 'check', 'load_historical', 'full-extract', 'all'], default='all')
    args = parser.parse_args()

    # --- LOAD CONFIGURATION ---
    project_dir = Path(__file__).parent
    load_dotenv(project_dir / "config" / "config.env")
    
    config = {
        "project_dir": project_dir,
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("POS_API_KEY"),
        "s3_bucket": os.getenv("S3_BUCKET_NAME")
    }
    if not all([config['base_url'], config['api_key'], config['s3_bucket']]):
        raise ValueError("One or more required environment variables are not set.")

    # --- RUN STEPS BASED ON ARGUMENT ---
    if args.step in ['extract', 'all']:
        raw_dir, file_tag = run_extract(config)
    
    if args.step in ['transform', 'all']:
        if 'raw_dir' not in locals(): # Ensure dependencies exist if running step alone
            file_tag = get_monthly_time_range()[0][:7]
            raw_dir = config['project_dir'] / "data" / "raw"
        transformed_df = run_transform(raw_dir, file_tag)
    
    if args.step in ['load', 'all']:
        if 'transformed_df' not in locals():
            file_tag = get_monthly_time_range()[0][:7]
            raw_dir = config['project_dir'] / "data" / "raw"
            transformed_df = run_transform(raw_dir, file_tag)
        run_load(transformed_df, config, file_tag)

    if args.step in ['check']:
        if 'state.json' not in locals():
            state_file_path = config['project_dir'] / "config" / "state.json" 
            if not state_file_path.exists():
                logging.warning("State file does not exist. Cannot check for new data.")
                return
        run_check(config)
    
    if args.step in ['load_historical']:
        run_load_historical_data(config)

    if args.step in ['cumulative_report', 'all']:
        run_cumulative_report(config)

    if args.step in ['monthly_report', 'all']:
        if 'file_tag' not in locals():
            file_tag = get_monthly_time_range()[0][:7]
        run_monthly_report(config, file_tag)

    if args.step in  ['full-extract']:
        run_extract_historical_data(config)

if __name__ == "__main__":
    main()