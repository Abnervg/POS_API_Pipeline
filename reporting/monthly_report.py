import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

# Import your data preparation functions
from .data_preparation import clean_data_for_reporting, explode_combo_items_advanced
from .data_preparation import calculate_beverage_distribution, calculate_mayo_percentages_and_counts, calculate_sales_by_day_of_week
from .data_preparation import calculate_daily_sales_metrics
from .data_preparation import calculate_sales_by_day_for_comparison

# Requests monthly data

def request_monthly_data(bucket_name):
    """
    Loads last and current monthly data from the S3 data lake to return a 2 month data dataframe

    Args:
        s3_bucket (str): The S3 bucket where the curated data is stored.

    Returns:
        pd.DataFrame: A single DataFrame containing 2 months worht of data.
    """
    logger = logging.getLogger(__name__)
    # Defines current month tag
    now = datetime.now()
    last_month = now - relativedelta(months=1)
    previous_month = last_month - relativedelta(months=1)

    # Define the base path for your partitioned data
    s3_paths_to_load = [f"s3://{bucket_name}/curated_data/year={last_month.year}/month={last_month.month:02d}/",
                        f"s3://{bucket_name}/curated_data/year={previous_month.year}/month={previous_month.month:02d}/"
                        ]
    all_dfs = []
    logger.info(f"Attempting to load data from S3 with path s3://{bucket_name}/curated_data/ for {last_month.year} year and {previous_month.month:02d},{last_month.month:02d} months")
    for path in s3_paths_to_load:
        try:
            logger.info(f"Loading data from {path}")
            monthly_df = pd.read_parquet(path)
            all_dfs.append(monthly_df)
        except FileNotFoundError:
            logger.warning(f"No data found at the specified path: {path}. This may be expected if no previous month data")
        except Exception as e:
            logger.error(f"An error occurred while loading monthly data from S3: {e}")
            raise
  
    # Combine the dataframes
    if not all_dfs:
        logger.warning("No data was found for the last two months")
        return pd.DataFrame()
    combined_df = pd.concat(all_dfs,ignore_index=True)

    logger.info(f"Successfully loaded a total of {len(combined_df)} records from the last two months.")
    return combined_df 

# --- Plotting Functions ---


def plot_beverage_distribution(df, output_dir):
    """
    Creates a stacked bar chart with custom, grouped legends for each
    beverage category.
    """
    logger = logging.getLogger(__name__)

    # 1. Get the prepared data
    data_for_plot = calculate_beverage_distribution(df)

    # 2. Pivot the data for plotting
    counts_pivot = data_for_plot.pivot(index='category', columns='item_name', values='count').fillna(0)
    percentages_pivot = data_for_plot.pivot(index='category', columns='item_name', values='percentage').fillna(0)

    # 3. Create the plot with the 'tab10' colormap
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.6, colormap='tab10')

    # 4. Add percentage labels by looking up the correct percentage value
    for container in ax.containers:
        # Get the item name for the current container (e.g., 'Coca Cola')
        item_name = container.get_label()
        
        labels = []
        for i, bar in enumerate(container):
            # Get the category name for the current bar (e.g., 'Refrescos y Aguas')
            category_name = counts_pivot.index[i]
            
            try:
                # Look up the percentage from the percentages_pivot DataFrame
                percentage = percentages_pivot.loc[category_name, item_name]
                if percentage > 5: # Only show label if segment is large enough
                    labels.append(f'{percentage:.0f}%')
                else:
                    labels.append('')
            except KeyError:
                # This handles cases where a combination doesn't exist
                labels.append('')
                
        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    # --- 5. Create Custom Grouped Legends ---
    handles, labels = ax.get_legend_handles_labels()
    item_to_category = pd.Series(data_for_plot.category.values, index=data_for_plot.item_name).to_dict()
    
    # Create a mapping of category to its legend items (handle and label)
    category_legends = {cat: [] for cat in data_for_plot['category'].unique()}
    for handle, label in zip(handles, labels):
        category = item_to_category.get(label)
        if category:
            category_legends[category].append((handle, label))
            
    # Remove the default legend
    if ax.get_legend():
        ax.get_legend().remove()

    # Add a new, custom legend for each category
    legend_y_start = 1.02 # Starting vertical position for the first legend
    for category, items in category_legends.items():
        if not items: continue
        
        # Unzip the handles and labels for the current category
        cat_handles, cat_labels = zip(*items)
        
        legend = ax.legend(cat_handles, cat_labels, title=f'-- {category} --', 
                           bbox_to_anchor=(1.02, legend_y_start), 
                           loc='upper left', 
                           title_fontsize='12',
                           fontsize='10')
        ax.add_artist(legend)
        legend_y_start -= 0.3 # Adjust this value to change spacing between legends

    # 6. Add titles and labels
    plt.title('Beverage Sales Distribution', fontsize=16)
    plt.xlabel('Beverage Category', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust layout to make space for legends

    # 7. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "beverage_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Beverage distribution plot saved to: {plot_path}")
 

def plot_stacked_counts_with_percentage_labels(df, output_dir):
    """Creates a stacked bar chart of Mayonesa modifiers."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_for_plot = calculate_mayo_percentages_and_counts(df)
    counts_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='count').fillna(0)
    percentages_pivot = data_for_plot.pivot(index='item_name', columns='mayo_type', values='percentage').fillna(0)

    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.7, colormap='tab10')

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

def plot_sales_by_day_of_week(df, output_dir):
    """
    Generates a line plot showing total sales traffic and traffic by order type.
    """
    logger = logging.getLogger(__name__)

    # 1. Get both total and categorized sales data
    total_sales, categorized_sales = calculate_sales_by_day_of_week(df)

    # 2. Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot the total line first, with a distinct style
    sns.lineplot(x=total_sales.index, y=total_sales.values, 
                 marker='o', color='black', linestyle='--', 
                 label='Total Sales', linewidth=2.5)

    # Plot the categorized lines using hue
    sns.lineplot(data=categorized_sales, x='day_of_week', y='count', 
                 hue='order_category', marker='o',
                 palette='tab10')

    # 3. Add titles and labels
    plt.title('Sales Traffic by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Unique Receipts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Order Type')
    plt.tight_layout()

    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "sales_by_weekday.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Sales by day of week plot saved to: {plot_path}")

def plot_daily_sales_trends(df, output_dir):
    """
    Generates a single-axis line plot of daily receipts with sales annotations.
    """
    logger = logging.getLogger(__name__)

    # 1. Get the daily metrics
    daily_data = calculate_daily_sales_metrics(df)

    # 2. Calculate the average ticket value for the whole month
    total_monthly_sales = daily_data['total_sales'].sum()
    total_monthly_receipts = daily_data['unique_receipts'].sum()
    avg_ticket_value = total_monthly_sales / total_monthly_receipts if total_monthly_receipts > 0 else 0

    # 3. Create the plot with a single Y-axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot unique receipts on the primary axis
    sns.lineplot(data=daily_data, x='date', y='unique_receipts', ax=ax, 
                 color='dodgerblue', marker='o', label='Unique Receipts')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Unique Receipts', fontsize=12)
    
    # 4. Add the total sales amount as a label on top of each data point
    for index, row in daily_data.iterrows():
        ax.text(row['date'], row['unique_receipts'] + 0.5, f"${row['total_sales']:,.0f}", 
                color='black', ha="center", va="bottom", fontsize=9)

    # 5. Add the average ticket value in the top-left corner
    avg_ticket_text = f"Avg. Ticket Value: ${avg_ticket_value:,.2f}"
    ax.text(0.02, 0.95, avg_ticket_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # 6. Add titles and formatting
    plt.title('Daily Customer Traffic and Sales', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    # 7. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "daily_sales_trends.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Daily sales trends plot saved to: {plot_path}")

# Plot monthly comparisons

def plot_monthly_comparison_by_weekday(df, output_dir):
    """
    Generates a comparative line plot showing sales traffic by day of the week
    for two different months.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly comparison plot for sales by weekday...")

    # 1. Get the prepared data
    comparison_data = calculate_sales_by_day_for_comparison(df)

    # 2. Create the plot
    plt.figure(figsize=(14, 8))
    
    # Use hue for the month and style for the order category
    ax = sns.lineplot(
        data=comparison_data,
        x='day_of_week',
        y='count',
        hue='month',          # Separates lines by month
        style='order_category', # Creates different dashing for order types
        palette=['gray', 'black'], # Sets the colors for each month
        markers=True,
        dashes=True,
        linewidth=2
    )

    # 3. Add titles and labels
    plt.title('Monthly Comparison of Sales Traffic by Day', fontsize=18)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Unique Receipts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Month & Order Type')
    plt.tight_layout()

    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "monthly_comparison_by_weekday.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Monthly comparison plot saved to: {plot_path}")




# --- Main Orchestrator Function for this Module ---

def generate_monthly_report(config):
    """
    Orchestrates the entire monthly report generation process.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Monthly Report Generation ---")
    
    # Request data
    bucket_name = config['s3_bucket']
    monthly_df = request_monthly_data(bucket_name)

    # 1. Prepare the data for reporting
    cleaned_df = clean_data_for_reporting(monthly_df)
    final_df = explode_combo_items_advanced(cleaned_df)
    
    # 2. Define output directory for this month's report
    now = datetime.now()
    report_output_dir = config['project_dir'] / "reports" / f"monthly_report_{now.year}_{now.month:02d}"
    
    # 3. Generate all plots

    #plot_stacked_counts_with_percentage_labels(final_df, report_output_dir)
    #plot_beverage_distribution(final_df, report_output_dir)
    #plot_sales_by_day_of_week(final_df, report_output_dir)
    #plot_daily_sales_trends(final_df, report_output_dir)
    plot_monthly_comparison_by_weekday(final_df, report_output_dir)

    # (Future step: Call a function to generate the .md summary file here)
    
    logger.info(f"--- Monthly Report generated successfully in: {report_output_dir} ---")
