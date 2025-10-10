import logging
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
import awswrangler as wr
import boto3

# Import all your custom functions at the top for clarity
from etl.extract import (
    fetch_incremental_data, 
    fetch_all_historical_data, 
    read_last_timestamp, 
    update_last_timestamp,
    save_raw_data
)
from etl.transform import run_transform # Assuming this is your main transform orchestrator
from etl.load import load_historical_data_from_local, merge_and_overwrite_monthly_data
from reporting.monthly_report import generate_monthly_report
from reporting.cumulative_report import generate_cumulative_report

# --- Main Orchestrator for the Automated Daily Run ---
def run_incremental_etl(config):
    """
    Orchestrates the complete daily incremental ETL process.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Daily Incremental ETL Run ---")

    state_file = config['project_dir'] / "config" / "etl_state.json"
    s3_bucket = config['s3_bucket']

    try:
        # 1. Read the last known timestamp from the state file
        last_timestamp = read_last_timestamp(state_file)

        # 2. Extract only new data since that timestamp
        new_receipts, new_items = fetch_incremental_data(
            config['base_url'], config['api_key'], last_timestamp
        )

        if not new_receipts:
            logger.info("No new receipts found. Pipeline finished for today.")
            return

        # 3. Transform only the new data
        new_df = run_transform(new_receipts, new_items)

        # 4. Merge the new data with historical data in S3
        merge_and_overwrite_monthly_data(new_df, s3_bucket)

        # 5. Update the state file with the newest timestamp from this batch
        update_last_timestamp(state_file, new_receipts)

        logger.info("--- Daily Incremental ETL Run Completed Successfully ---")

    except Exception as e:
        logger.error(f"Incremental ETL pipeline failed: {e}")
        # The state file is NOT updated on failure, ensuring we retry from the same point
        raise

    # 6. Repair Athena table to recognize any new partitions
    session = boto3.Session(region_name='us-east-1')  # Adjust region as needed

    try:
        logger.info("Repairing table to discover any new partitions...")
        wr.athena.repair_table(
            table="sales_data_appended",
            database="your_athena_database_name",
            boto3_session=session
        )
        logger.info("Table repair successful.")
    except Exception as e:
        logger.error(f"Failed to repair Athena table: {e}")

# --- Orchestrators for Manual / One-Time Tasks ---

def run_full_historical_extract(config):
    """Extracts all historical data from the API and saves it locally."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Full Historical Data Extraction ---")
    
    all_receipts, all_items = fetch_all_historical_data(config['base_url'], config['api_key'])
    
    output_dir = config['project_dir'] / "data" / "raw"
    date_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
    file_tag = f"historical_extract_{date_str}"
    save_raw_data(all_receipts, all_items, output_dir, file_tag)
    
    logger.info("--- Finished Full Historical Data Extraction ---")


def main():
    # --- CONFIGURE LOGGING ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run the ETL and Reporting pipeline.")
    parser.add_argument(
        '--step', 
        choices=['daily_run', 'full_extract', 'load_historical', 'monthly_report', 'cumulative_report', 'report'], 
        required=True,
        help='Specify which pipeline step to run.'
    )
    args = parser.parse_args()

    # --- LOAD CONFIGURATION ---
    project_dir = Path(__file__).parent
    load_dotenv(project_dir / "config" / "config.env")
    
    config = {
        "project_dir": project_dir,
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("POS_API_KEY"),
        "s3_bucket": os.getenv("S3_BUCKET_NAME"),
        "recipient_email": os.getenv("RECIPIENT_EMAIL"),
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": os.getenv("SMTP_PORT"),
        "athena_database": os.getenv("ATHENA_DATABASE"),
        "athena_table": os.getenv("ATHENA_TABLE")
    }
    if not all(config.values()):
        raise ValueError("One or more required environment variables are not set.")

    # --- RUN STEPS BASED ON ARGUMENT ---
    if args.step == 'daily_run':
        run_incremental_etl(config)
    elif args.step == 'full_extract':
        run_full_historical_extract(config)
    elif args.step == 'load_historical':
        load_historical_data_from_local(config['project_dir'] / "data" / "raw", config['project_dir'] / "config" / "etl_state.json", config['s3_bucket'])
    elif args.step == 'monthly_report':
        generate_monthly_report(config)
    elif args.step == 'cumulative_report':
        generate_cumulative_report(config)
    elif args.step == 'report':
        generate_cumulative_report(config)
        generate_monthly_report(config)

if __name__ == "__main__":
    main()