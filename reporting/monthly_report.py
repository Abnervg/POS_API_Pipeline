import pandas as pd

#Top 5 products by sales this month
def get_top_sold_items(df, top_n=5):
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

# (Other imports)
from .data_preparation import clean_data_for_reporting, explode_combo_items_advanced
# from etl.load import load_curated_data # Function to load your data

def generate_monthly_report(monthly_df):
    
    # PERFORM initial cleaning
    cleaned_df = clean_data_for_reporting(monthly_df)
    
    # EXPLODE the combo items
    final_df = explode_combo_items_advanced(cleaned_df)
    
    # START ANALYSIS on the final, fully-cleaned data
    
    # Get top 5 products by sales
    top_products = top_five_products(final_df)
    return top_products
