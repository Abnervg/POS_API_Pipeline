import logging
import argparse
from pathlib import Path
import os
from venv import logger
from dotenv import load_dotenv
import pandas as pd

# Import custom ETL functions
from etl.extract import get_monthly_time_range, fetch_api_data, save_raw_data
from etl.transform import load_raw_data, flattening_table_mine, homogenize_order_types, time_slots
from etl.load import load_to_curated_folder,load_to_aws_bucket

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

def run_load(processed_df, file_tag):
    """Runs the entire data loading process."""
    logger = logging.getLogger(__name__)
    curated_folder = Path(__file__).parent / "data" / "curated"
    curated_folder.mkdir(parents=True, exist_ok=True)   
    logger.info("--- Starting Load Step to curated folder---")
    load_to_curated_folder(processed_df, file_tag)
    logger.info("--- Finished Load Step to curated folder ---")
    logger.info("--- Starting Load Step to AWS S3 bucket ---")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    load_to_aws_bucket(processed_df, bucket_name, file_tag)
    logger.info("--- Finished Load Step to AWS S3 bucket ---")

def get_monthly_report():
    """
    Generates the monthly report by cleaning, exploding combo items, and analyzing the data.
    Returns the top products DataFrame.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Monthly Report Generation ---")
    
    # Load the raw data
    raw_data_dir = Path(__file__).parent / "data" / "raw"
    print(f"Raw data directory: {raw_data_dir}")
    file_tag = get_monthly_time_range()[0][:7]  # Use YYYY-MM as the tag
    monthly_df = load_raw_data(raw_data_dir, file_tag)
    
    # Generate the report
    from reporting.monthly_report import generate_monthly_report
    top_products = generate_monthly_report(monthly_df)
    
    logger.info("--- Finished Monthly Report Generation ---")
    return top_products

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
        choices=['extract', 'transform', 'load', 'report', 'all'], 
        default='all',
        help='Specify which step to run.'
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
        run_load(transformed_df,file_tag)
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
        try:
            s3_bucket_name = os.getenv("S3_BUCKET_NAME")
            if not s3_bucket_name:
                raise ValueError("S3_BUCKET_NAME must be set in the environment.")
        except Exception as e:
            logger.error(f"Error retrieving S3 bucket name: {e}")
            return

        transformed_df = run_transform(raw_dir, file_tag)
        curated_dir = project_dir / "data" / "curated"
        if not curated_dir.exists():    
            curated_dir.mkdir(parents=True, exist_ok=True)
        run_load(transformed_df, file_tag)
        load_to_aws_bucket(transformed_df, s3_bucket_name, file_tag)
    elif args.step == 'report':
        logger.info("--- Starting Monthly Report Generation ---")
        
        # This step assumes the curated data already exists.
        # It loads the final output of your transform/load steps.
        project_dir = Path(__file__).parent
        print(f"Project directory: {project_dir}")
        curated_dir = project_dir / "data" / "curated"
        file_tag = get_monthly_time_range()[0][:7] # Use YYYY-MM as the tag
        
        # Load the final, curated data, not the raw data
        curated_file_path = curated_dir / f"curated_data_{file_tag}.csv"
        if not curated_file_path.exists():
            raise FileNotFoundError(f"Curated data file not found for tag {file_tag} in dir {curated_file_path}. Please run the transform/load steps first.")

        final_df = pd.read_csv(curated_file_path)
        
        # Now, call your report generation function with the clean data
        from reporting.monthly_report import generate_monthly_report
        generate_monthly_report(final_df)

        logger.info("--- Finished Monthly Report Generation ---")

if __name__ == "__main__":
    main()