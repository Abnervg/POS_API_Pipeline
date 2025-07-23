import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns

# Import your data preparation functions
from .data_preparation import clean_data_for_reporting, explode_combo_items_advanced, calculate_mayo_percentages_and_counts, calculate_beverage_distribution

# --- Data Calculation Functions ---

def get_top_products(df, top_n=5):
    """Counts how many times each item appears and returns the top N items."""
    top_items_series = df['item_name'].value_counts().head(top_n)
    top_items_df = top_items_series.reset_index()
    top_items_df.columns = ['item_name', 'items_sold']
    return top_items_df

# --- Plotting Functions ---

def create_top_products_plot(df, output_dir):
    """Generates a bar chart of the top 5 sold products and saves it."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    top_products_df = get_top_products(df, top_n=5)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_products_df, x='item_name', y='items_sold', palette='viridis')
    plt.title('Top 5 Most Sold Items', fontsize=16)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = output_dir / "top_5_products.png"
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory
    logger.info(f"Top products plot saved to: {plot_path}")

def plot_beverage_distribution(df, output_dir):
    """
    Creates a stacked bar chart showing the distribution of beverages,
    with counts on the Y-axis and percentage labels inside the bars.
    """
    logger = logging.getLogger(__name__)

    # 1. Get the prepared data with counts and percentages
    data_for_plot = calculate_beverage_distribution(df)

    # 2. Pivot the data for both counts (for bar heights) and percentages (for labels)
    counts_pivot = data_for_plot.pivot(index='category', columns='item_name', values='count').fillna(0)
    percentages_pivot = data_for_plot.pivot(index='category', columns='item_name', values='percentage').fillna(0)

    # 3. Create the stacked bar plot using the raw counts
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(10, 8), width=0.6, colormap='viridis')

    # 4. Add the percentage labels to each segment
    for container in ax.containers:
        # Create labels from the corresponding percentage data
        labels = []
        for i, v in enumerate(container):
            category_name = counts_pivot.index[i]
            item_name = container.get_label()
            try:
                percentage = percentages_pivot.loc[category_name, item_name]
                if percentage > 5: # Only show label if segment is large enough
                    labels.append(f'{percentage:.0f}%')
                else:
                    labels.append('')
            except KeyError:
                labels.append('')
        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    # 5. Add titles and labels
    plt.title('Beverage Sales Distribution', fontsize=16)
    plt.xlabel('Beverage Category', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Beverage Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 6. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "beverage_distribution.png"
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory
    
    logger.info(f"Beverage distribution plot saved to: {plot_path}")
 

def plot_stacked_counts_with_percentage_labels(df, output_dir):
    """Creates a stacked bar chart of Mayonesa modifiers."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_for_plot = calculate_mayo_percentages_and_counts(df)
    counts_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='count').fillna(0)
    percentages_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='percentage').fillna(0)

    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.7, colormap='plasma')

    for container in ax.containers:
        labels = []
        for i, v in enumerate(container):
            burger_name = counts_pivot.index[i]
            mayo_type = container.get_label()
            try:
                percentage = percentages_pivot.loc[burger_name, mayo_type]
                if percentage > 0:
                    labels.append(f'{percentage:.1f}%')
                else:
                    labels.append('')
            except KeyError:
                labels.append('')
        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    plt.title('Mayonnaise Preference per Burger Type', fontsize=16)
    plt.ylabel('Number of Burgers Sold', fontsize=12)
    plt.xlabel('Burger Type', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Mayo Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_path = output_dir / "mayonnaise_stacked_counts.png"
    plt.savefig(plot_path)
    plt.close() # Close the plot
    logger.info(f"Stacked count plot saved to: {plot_path}")

# --- Main Orchestrator Function for this Module ---

def generate_monthly_report(df, config, file_tag):
    """
    Orchestrates the entire monthly report generation process.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Monthly Report Generation ---")
    
    # 1. Prepare the data for reporting
    cleaned_df = clean_data_for_reporting(df)
    final_df = explode_combo_items_advanced(cleaned_df)
    
    # 2. Define output directory for this month's report
    report_output_dir = config['project_dir'] / "reports" / f"monthly_report_{file_tag}"
    
    # 3. Generate all plots

    plot_stacked_counts_with_percentage_labels(final_df, report_output_dir)
    plot_beverage_distribution(final_df, report_output_dir)
    
    # (Future step: Call a function to generate the .md summary file here)
    
    logger.info(f"--- Monthly Report generated successfully in: {report_output_dir} ---")
