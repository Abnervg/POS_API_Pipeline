import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path

#Top 5 products by sales this month
def get_top_products(df, top_n=5):
    """
    Counts how many times each item appears and returns the top N items.
    """
    # .value_counts() is the perfect tool for this. It counts and sorts automatically.
    top_items_series = df['item_name'].value_counts().head(top_n)
    
    # Convert the Series to a DataFrame for easier handling
    top_items_df = top_items_series.reset_index()
    
    # Rename the columns for clarity
    top_items_df.columns = ['item_name', 'items_sold']
    
    return top_items_df


def create_top_products_plot(df, output_dir):
    """
    Generates a bar chart of the top 5 sold products and saves it to a file.

    Args:
        df (pd.DataFrame): The cleaned DataFrame for analysis.
        output_dir (Path): The directory where the plot image will be saved.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Get the top 5 items
    top_products_df = get_top_products(df, top_n=5)
    
    # 3. Create the plot
    plt.figure(figsize=(10, 7)) # Create a figure with a nice size
    plt.bar(top_products_df['item_name'], top_products_df['items_sold'], color='teal')
    
    # 4. Add labels and a title for clarity
    plt.title('Top 5 Most Sold Items', fontsize=16)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # 5. Save the plot to a file inside the specified directory
    plot_path = output_dir / "top_5_products.png"
    plt.savefig(plot_path)
    
    logger.info(f"Top products plot saved to: {plot_path}")

from .data_preparation import clean_data_for_reporting, explode_combo_items_advanced
# from etl.load import load_curated_data # Function to load your data

def generate_monthly_report(monthly_df):
    
    # PERFORM initial cleaning
    cleaned_df = clean_data_for_reporting(monthly_df)
    
    # EXPLODE the combo items
    final_df = explode_combo_items_advanced(cleaned_df)
    
    # START ANALYSIS on the final, fully-cleaned data
    
    # Get top 5 products by sales
    top_products = get_top_products(final_df)
    # Create a plot of the top products
    output_dir = Path(__file__).parent / "plots"
    create_top_products_plot(final_df, output_dir)
    return top_products
