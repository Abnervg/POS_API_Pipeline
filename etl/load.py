import logging
from pathlib import Path
import pandas as pd
import json

#Import transform functions
from etl.transform import homogenize_order_types, time_slots, flattening_table_mine
from etl.extract import update_last_timestamp

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

# ==============================================================================
# The NEW, UNIFIED function for saving data to S3
# ==============================================================================
def save_to_s3_partitioned(df_to_save, s3_bucket):
    """
    Saves a DataFrame to S3, partitioning it by year and month with a
    single 'data.parquet' file in each monthly folder. This function enforces
    a consistent schema to prevent read errors.
    """
    logger = logging.getLogger(__name__)
    
    if df_to_save.empty:
        logger.warning("Input DataFrame is empty. Nothing to save.")
        return

    # --- Ensure a consistent schema before saving ---
    df_to_save['shifted_time'] = pd.to_datetime(df_to_save['shifted_time'], errors='coerce')
    for col in df_to_save.select_dtypes(include=['object']).columns:
        df_to_save[col] = df_to_save[col].astype('string')

    # Create a 'year_month' helper column
    df_to_save['year_month'] = df_to_save['shifted_time'].dt.strftime('%Y-%m')

    # Loop through each month and save it as a single file
    unique_months = df_to_save['year_month'].unique()
    logger.info(f"Saving data for the following months: {unique_months}")

    for month_tag in unique_months:
        year, month = month_tag.split('-')
        
        monthly_data = df_to_save[df_to_save['year_month'] == month_tag].copy()
        monthly_data.drop(columns=['year_month'], inplace=True)

        # Define the S3 path for this month's file (partitioned by folder)
        s3_path_for_month = f"s3://{s3_bucket}/curated_data/year={year}/month={month}/data.parquet"
        
        logger.info(f"Uploading {len(monthly_data)} records for {month_tag} to {s3_path_for_month}")
        monthly_data.to_parquet(s3_path_for_month, index=False)

    logger.info("Finished saving all partitioned data to S3.")


# ==============================================================================
# The function to orchestrate the one-time historical load
# ==============================================================================

def load_historical_data_from_local(local_raw_dir, etl_state_file_dir, s3_bucket):
    """
    Loads all historical raw JSON data from a local folder, transforms it,
    and then calls the unified save function.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting historical data load from {local_raw_dir} ---")

    # 1. Load all raw JSON files from the local directory
    all_receipts = []
    all_items = []
    
    receipt_files = list(local_raw_dir.glob("receipts_*.json"))
    items_files = list(local_raw_dir.glob("items_*.json"))

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
    
    # 3. The final step is to call the new, unified save function
    save_to_s3_partitioned(historical_df, s3_bucket)

    # 4. Find the latest timestamp from the raw data and update the state file
    if all_receipts:
        state_file_path = etl_state_file_dir
        update_last_timestamp(state_file_path, all_receipts)
    logger.info("--- Finished historical data load ---")


# ==============================================================================
# The function for the daily incremental update
# ==============================================================================


def merge_and_overwrite_monthly_data(new_df, s3_bucket):
    """
    Loads monthly data from S3, merges new data, and overwrites the files.
    This function can handle new data that spans multiple months.
    """
    logger = logging.getLogger(__name__)
    
    if new_df.empty:
        logger.info("No new data to load. Skipping merge process.")
        return

    # Prepare the new DataFrame
    new_df['shifted_time'] = pd.to_datetime(new_df['shifted_time'], errors='coerce')
    new_df.dropna(subset=['shifted_time'], inplace=True)

    # Find all unique months in the new data
    new_df['year_month'] = new_df['shifted_time'].dt.strftime('%Y-%m')
    unique_months_in_new_data = new_df['year_month'].unique()

    logger.info(f"New data spans the following months: {unique_months_in_new_data}")

    # Process each month
    for month_tag in unique_months_in_new_data:
        year, month = month_tag.split('-')
        monthly_data_to_add = new_df[new_df['year_month'] == month_tag].copy()

        # Path for this month's parquet file
        s3_path = f"s3://{s3_bucket}/curated_data/year={year}/month={month}/data.parquet"
        logger.info(f"Updating data for {month_tag} at path: {s3_path}")

        try:
            # Load existing monthly parquet
            historical_df = pd.read_parquet(s3_path)
            logger.info(f"Loaded {len(historical_df)} existing records for {month_tag}.")
            combined_df = pd.concat([historical_df, monthly_data_to_add], ignore_index=True)
        except FileNotFoundError:
            logger.info(f"No existing data found for {month_tag}. Starting with new data.")
            combined_df = monthly_data_to_add

        # Deduplicate
        combined_df.sort_values(by='shifted_time', ascending=False, inplace=True)
        combined_df.drop_duplicates(subset=['receipt_number', 'item_name'], keep='first', inplace=True)

        # Normalize object columns to string
        for col in combined_df.select_dtypes(include=['object']).columns:
            combined_df[col] = combined_df[col].astype('string')
        
        # Drop helper column
        final_df_to_save = combined_df.drop(columns=['year_month'])

        # Save back to S3
        logger.info(f"Uploading {len(final_df_to_save)} total records for {month_tag} to S3.")
        final_df_to_save.to_parquet(s3_path, index=False)

    logger.info("Finished merging and loading all new data to S3.")