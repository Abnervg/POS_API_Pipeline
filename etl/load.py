import logging
from pathlib import Path
import pandas as pd

#Define output directory for curated data
app_path = Path(__file__).parent
output_dir = app_path.parent / "data" / "curated"

def load_to_curated_folder(flat_table, file_tag, output_dir=output_dir):
    """
    Load the transformed data into a curated folder.
    
    :param flat_table: The DataFrame containing the transformed data.
    :param output_dir: The directory where the curated data will be saved.
    :param file_tag: A tag to identify the file, typically based on the date.
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
    from io import StringIO
    
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    flat_table.to_csv(csv_buffer, index=False)

    logging.info(f"Uploading curated data to S3 bucket {bucket_name} with key curated_data_{file_tag}.csv")
    s3.put_object(Bucket=bucket_name, Key=f"curated_data_{file_tag}.csv", Body=csv_buffer.getvalue())
    logging.info(f"Curated data uploaded to S3 bucket {bucket_name} with key curated_data_{file_tag}.csv")