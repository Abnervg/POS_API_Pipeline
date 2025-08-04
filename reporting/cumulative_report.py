 #This file produces a cumulative report of the data in the database.
# It aggregates data from various tables and generates a summary report.

import pandas as pd
import boto3
from io import BytesIO
import logging

def request_data(bucket_name):
    """
    Loads all partitioned monthly data from S3 and merges them into a single dataframe for cumulative analysis

        Args:bucketname(str): S3 bucketname where data is stored

        Returns:  
    """
    logger = logging.getLogger(__name__)
    logging.info(f"Downloading historical data from S3 bucket {bucket_name}")
    s3_path = f"s3://{bucket_name}/curated_data/**/*.parquet"

    #Merge data

    try:
        historical_df = pd.read_parquet(s3_path)
        logger.info(f"Succesfully extracted data from S3 {bucket_name}")

