import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os

#Define paths
app_path = Path(__file__).parent
output_dir = app_path / "data" / "curated"

def load_to_curated_folder(flat_table,output_dir, file_tag):
    """
    Load the transformed data into a curated folder.
    
    :param flat_table: The DataFrame containing the transformed data.
    :param output_dir: The directory where the curated data will be saved.
    :param file_tag: A tag to identify the file, typically based on the date.
    """
    curated_file_path = output_dir / f"curated_data_{file_tag}.csv"
    flat_table.to_csv(curated_file_path, index=False)
    logging.info(f"Curated data saved to {curated_file_path}")