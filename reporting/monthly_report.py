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



def calculate_mayo_percentages_and_counts(df):
    """
    Calculates both the raw count and the percentage distribution of 
    Mayonesa modifiers for each burger type.

    Returns:
        pd.DataFrame: A DataFrame with columns ['item_name', 'mayo_type', 'count', 'percentage']
    """
    # Filter for burgers and then for Mayonesa modifiers
    all_burgers = df[df['item_name'].str.contains("Burger|Smash", case=False, na=False)].copy()
    mayo_burgers = all_burgers[all_burgers['modifiers'].str.contains("Mayonesa", case=False, na=False)].copy()
    
    # Extract the specific mayo type
    mayo_burgers['mayo_type'] = mayo_burgers['modifiers'].str.extract(r'Mayonesa\((.*?)\)')
    
    # Get the count of each mayo type per burger
    mayo_counts = mayo_burgers.groupby(['item_name', 'mayo_type']).size().reset_index(name='count')

    # Get the total number of mayo burgers for each type to calculate percentages
    total_burgers_per_item = mayo_counts.groupby('item_name')['count'].transform('sum')
    mayo_counts['percentage'] = (mayo_counts['count'] / total_burgers_per_item) * 100
    
    return mayo_counts

def plot_stacked_counts_with_percentage_labels(df, output_dir):
    """
    Creates a stacked bar chart showing the raw count of each Mayonesa modifier,
    with percentage labels inside each bar segment.
    """
    logger = logging.getLogger(__name__)

    # 1. Get the data with both counts and percentages calculated
    data_for_plot = calculate_mayo_percentages_and_counts(df)

    # 2. Pivot the data for both counts (for bar heights) and percentages (for labels)
    counts_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='count').fillna(0)
    percentages_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='percentage').fillna(0)

    # 3. Create the stacked bar plot using the raw counts
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.7)

    # 4. Add the percentage labels to each segment
    for container in ax.containers:
        # Create labels from the corresponding percentage data
        # We need to get the labels from the pivoted percentage table
        labels = []
        for i, v in enumerate(container):
            # Get the burger name and mayo type for this bar segment
            burger_name = v.get_x() + v.get_width() / 2
            mayo_type = container.get_label()
            
            # Find the corresponding percentage
            try:
                percentage = percentages_pivot.loc[counts_pivot.index[i], mayo_type]
                if percentage > 0:
                    labels.append(f'{percentage:.1f}%')
                else:
                    labels.append('')
            except KeyError:
                labels.append('')

        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    # 5. Add titles and labels
    plt.title('Mayonnaise Preference per Burger Type (by Volume)', fontsize=16)
    plt.xlabel('Burger Type', fontsize=12)
    plt.ylabel('Number of Burgers Sold', fontsize=12) # Y-axis now shows count
    plt.xticks(rotation=0)
    plt.legend(title='Mayo Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 6. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "mayonnaise_stacked_counts_with_percent_labels.png"
    plt.savefig(plot_path)
    
    logger.info(f"Stacked count plot with percentage labels saved to: {plot_path}")


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
    plot_stacked_counts_with_percentage_labels(final_df, output_dir)
    return top_products
