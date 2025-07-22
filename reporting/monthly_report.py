import pandas as pd

#Top 5 products by sales this month
def top_five_products(df):
    """
    Returns the top 5 products by sales as a DataFrame.
    """
    # Group by item, sum the price, and get the top 5
    top_products_series = df.groupby('item_name')['price'].sum().nlargest(5)
    
    # Convert the resulting Series to a DataFrame
    top_products_df = top_products_series.reset_index()
    
    # Rename the columns to match what the test expects
    top_products_df.columns = ['item_name', 'total_sales']

    return top_products_df

# (Other imports)
from .data_preparation import clean_data_for_reporting, explode_combo_items
# from etl.load import load_curated_data # Function to load your data

def generate_monthly_report(monthly_df):
    
    # PERFORM initial cleaning
    cleaned_df = clean_data_for_reporting(monthly_df)
    
    # EXPLODE the combo items
    final_df = explode_combo_items(cleaned_df)
    
    # START ANALYSIS on the final, fully-cleaned data
    
    # Get top 5 products by sales
    top_products = top_five_products(final_df)
    return top_products
