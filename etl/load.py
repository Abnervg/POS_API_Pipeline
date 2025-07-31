import logging
from pathlib import Path
import pandas as pd
import json

#Import transform functions
from etl.transform import homogenize_order_types, time_slots, flattening_table_mine

#Define output directory for curated data
app_path = Path(__file__).parent
output_dir = app_path.parent / "data" / "curated"

def load_to_curated_folder(flat_table, output_dir, file_tag):
    """
    Load the transformed data into a curated folder.
    
    :param flat_table: The DataFrame containing the transformed data.
    :param output_dir: The Path object for the directory where the data will be saved.
    :param file_tag: A string tag to identify the file (e.g., '2025-06').
    """
    curated_file_path = output_dir / f"curated_data_{file_tag}.csv"
    flat_table.to_csv(curated_file_path, index=False)
    logging.info(f"Curated data saved to {curated_file_path}")

def load_to_aws_bucket(flat_table, bucket_name, file_tag):
    """
    Load the transformed data into an AWS S3 bucket.
    
    :param flat_table: The DataFrame containing the transformed data.
    :param bucket_name: The name of the S3 bucket.
    :param file_tag: A tag to identify the file, typically based on the date.
    """
    import boto3
    from io import BytesIO
    
    s3 = boto3.client('s3')
    parquet_buffer = BytesIO()
    flat_table.to_parquet(parquet_buffer, index=False)

    logging.info(f"Uploading curated data to S3 bucket {bucket_name} with key curated_data_{file_tag}.parquet")
    s3.put_object(Bucket=bucket_name, Key=f"curated_data_{file_tag}.parquet", Body=parquet_buffer.getvalue())
    logging.info(f"Curated data uploaded to S3 bucket {bucket_name} with key curated_data_{file_tag}.parquet")

def load_historical_data_from_local(local_raw_dir, s3_bucket):
    """
    A one-time function to load all historical raw JSON data from a local
    folder, transform it, and merge it into the partitioned S3 data lake.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting historical data backfill from {local_raw_dir} ---")

    # 1. Find and load all raw JSON files from the local directory
    all_receipts = []
    all_items = []
    
    receipt_files = list(local_raw_dir.glob("receipts_*.json"))
    items_files = list(local_raw_dir.glob("items_*.json")) # Assuming one items file for simplicity

    if not receipt_files or not items_files:
        logger.error("No raw receipt or item files found in the specified directory.")
        return

    for file_path in receipt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            all_receipts.extend(json.load(f))

    with open(items_files[0], "r", encoding="utf-8") as f:
        all_items = json.load(f)
    
    logger.info(f"Loaded a total of {len(all_receipts)} historical receipts.")

    # 2. Transform the entire historical dataset
    logger.info("Transforming historical data...")
    historical_df = flattening_table_mine((all_receipts, all_items))
    historical_df = homogenize_order_types(historical_df)
    historical_df = time_slots(historical_df)
    logger.info("Transformation of historical data complete.")

    # 3. Call the existing partitioned load function to merge and upload
    # This reuses your existing logic to handle the partitioning and deduplication.
    merge_and_load_partitioned_data(historical_df, s3_bucket)

    logger.info("--- Historical data backfill complete. ---")

def merge_and_load_partitioned_data(new_df, s3_bucket):
    """
    Merges new data with historical data in S3, partitioned by year and month.
    """
    logger = logging.getLogger(__name__)
    
    if new_df.empty:
        logger.info("No new data to load. Skipping merge process.")
        return

    # Ensure the datetime column is in the correct format to extract year and month
    new_df['shifted_time'] = pd.to_datetime(new_df['shifted_time'], errors='coerce')
    new_df.dropna(subset=['shifted_time'], inplace=True)

    # Find the unique months in the new data (e.g., ['2025-06', '2025-07'])
    new_df['year_month'] = new_df['shifted_time'].dt.strftime('%Y-%m')
    unique_months = new_df['year_month'].unique()

    logger.info(f"New data spans the following months: {unique_months}")

    # Process each month's data separately
    for month_tag in unique_months:
        year, month = month_tag.split('-')
        
        # Filter the new data for the current month being processed
        monthly_data_to_add = new_df[new_df['year_month'] == month_tag]

        # Define the S3 path for this specific month's data
        s3_path = f"s3://{s3_bucket}/curated_data/year={year}/month={month}/data.parquet"
        logger.info(f"Processing data for {month_tag} at path: {s3_path}")

        try:
            # 1. Load the existing historical data for this month from S3
            historical_df = pd.read_parquet(s3_path)
            logger.info(f"Loaded {len(historical_df)} historical records for {month_tag}.")
        except FileNotFoundError:
            # If the file doesn't exist, it's the first run for this month
            historical_df = pd.DataFrame()
            logger.info(f"No historical data found for {month_tag}. Creating new file.")

        # 2. Append the new data
        combined_df = pd.concat([historical_df, monthly_data_to_add], ignore_index=True)

        # 3. Deduplicate, keeping the most recent entry for each receipt
        # This is crucial for data integrity
        combined_df.sort_values(by='shifted_time', ascending=False, inplace=True)
        combined_df.drop_duplicates(subset=['receipt_number', 'item_name'], keep='first', inplace=True)

        # 4. Save the merged and deduplicated data back to S3, overwriting the old file
        logger.info(f"Uploading {len(combined_df)} total records for {month_tag} to S3.")
        combined_df.to_parquet(s3_path, index=False)

    logger.info("Finished merging and loading all new data to S3.")