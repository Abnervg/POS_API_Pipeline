import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns

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



def calculate_mayo_percentages(df):
    """
    Calculates the percentage distribution of Mayonesa modifiers for each burger type.
    """
    # Start with the existing count logic
    all_burgers = df[df['item_name'].str.contains("Burger|Smash", case=False, na=False)].copy()
    mayo_burgers = all_burgers[all_burgers['modifiers'].str.contains("Mayonesa", case=False, na=False)].copy()
    mayo_burgers['mayo_type'] = mayo_burgers['modifiers'].str.extract(r'Mayonesa\((.*?)\)')
    
    # Get the count of each mayo type per burger
    mayo_counts = mayo_burgers.groupby(['item_name', 'mayo_type']).size().reset_index(name='count')

    # Get the TOTAL number of mayo burgers for each type
    total_burgers = mayo_counts.groupby('item_name')['count'].sum().reset_index(name='total_count')

    # Merge the total back to calculate percentages
    percentage_df = pd.merge(mayo_counts, total_burgers, on='item_name')
    percentage_df['percentage'] = (percentage_df['count'] / percentage_df['total_count']) * 100
    
    return percentage_df

def plot_stacked_mayo_percentages(df, output_dir):
    """
    Creates a 100% stacked bar chart showing the percentage of each
    Mayonesa modifier per burger type, with labels.
    """
    logger = logging.getLogger(__name__)

    # 1. Get the data with percentages calculated
    percentage_df = calculate_mayo_percentages(df)

    # 2. Pivot the data to prepare for stacking
    plot_data = percentage_df.pivot(
        index='item_name', 
        columns='mayo_type', 
        values='percentage'
    ).fillna(0)

    # 3. Create the stacked bar plot
    ax = plot_data.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.7)

    # 4. Add the percentage labels to each segment of the bars
    for container in ax.containers:
        # Format labels to show one decimal place and a '%' sign
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container]
        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    # 5. Add titles and labels
    plt.title('Mayonnaise Preference per Burger Type', fontsize=16)
    plt.xlabel('Burger Type', fontsize=12)
    plt.ylabel('Percentage of Sales', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Mayo Type', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # 6. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "mayonnaise_percentage_by_burger.png"
    plt.savefig(plot_path)
    
    logger.info(f"Mayonnaise percentage plot saved to: {plot_path}")

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
    # Plot burgers with modifiers
    plot_stacked_mayo_percentages(final_df, output_dir)
    return top_products
