 #This file produces a cumulative report of the data in the database.
# It aggregates data from various tables and generates a summary report.

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

#Import cleaning functions
from reporting.data_preparation import clean_data_for_reporting, explode_combo_items_advanced



# Define transformation functions to be used in the report generation
def calculate_cumulative_metrics(df):
    """
    Calculate cumulative metrics from the DataFrame.
    
    :param df: The DataFrame containing the data.
    :return: A DataFrame with cumulative KPIs.
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating cumulative metrics from the DataFrame.")

    # Ensure 'price' is numeric and 'shifted_time' is datetime
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # Calculate KPIs
    total_revenue = df['price'].sum()
    total_receipts = df['receipt_number'].nunique()
    average_receipt_value = total_revenue / total_receipts if total_receipts > 0 else 0
    first_sale_date = df['shifted_time'].min().date()
    last_sale_date = df['shifted_time'].max().date()
    average_monthly_revenue = df.groupby(df['shifted_time'].dt.to_period('M'))['price'].sum().mean()
    
    kpis = {
        "Total Revenue": f"${total_revenue:,.2f}",
        "Total Unique Receipts": f"{total_receipts:,}",
        "Average Receipt Value": f"${average_receipt_value:,.2f}",
        "Average Monthly Revenue": f"${average_monthly_revenue:,.2f}",
        "First Sale Date": first_sale_date.strftime('%Y-%m-%d'),
        "Last Sale Date": last_sale_date.strftime('%Y-%m-%d')
    }
    
    return kpis

# Define plotting functions to visualize the data
def plot_cumulative_sales_trend(df, output_dir):
    """
    Generates a time series line plot of total sales per month.

    Args:
        df (pd.DataFrame): The complete historical data.
        output_dir (Path): The directory where the plot will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating cumulative sales trend plot...")

    df = df.copy()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # Resample the data to get the sum of sales for each month
    monthly_sales = df.set_index('shifted_time')['price'].resample('M').sum()

    # Create the plot
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(
        x=monthly_sales.index, 
        y=monthly_sales.values, 
        marker='o', 
        linestyle='-',
        color='navy'
    )

    # Add titles and labels
    plt.title('Total Sales Trend Over Time', fontsize=18)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Format the y-axis to show dollar values
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "cumulative_sales_trend.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Cumulative sales trend plot saved to: {plot_path}")

# Function to load data from S3

def request_data(bucket_name):
    """
    Loads all partitioned monthly data from the S3 data lake and combines it
    into a single DataFrame for cumulative analysis.

    Args:
        s3_bucket (str): The S3 bucket where the curated data is stored.

    Returns:
        pd.DataFrame: A single DataFrame containing all historical data.
    """
    logger = logging.getLogger(__name__)
    
    # Define the base path for your partitioned data
    base_s3_path = f"s3://{bucket_name}/curated_data/"
    
    logger.info(f"Loading all historical data from S3 path: {base_s3_path}")
    
    try:
        # Use pandas to read the entire partitioned dataset.
        # It will automatically discover the 'year=' and 'month=' directories.
        historical_df = pd.read_parquet(base_s3_path)
        
        logger.info(f"Successfully loaded a total of {len(historical_df)} historical records.")
        return historical_df
        
    except FileNotFoundError:
        logger.warning(f"No data found at the specified path: {base_s3_path}. Returning an empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred while loading historical data from S3: {e}")
        raise

# Cumulative report template generation function
def create_cumulative_summary_report(df, exploded_df, output_dir):
    """
    Generates a comprehensive cumulative summary report in Markdown format.

    Args:
        df (pd.DataFrame): The complete historical data, cleaned but not exploded.
        exploded_df (pd.DataFrame): The historical data after exploding combo items.
        output_dir (Path): The directory to save the report file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive cumulative summary report...")

    # --- 1. Calculate Key Performance Indicators (KPIs) ---
    # Call the dedicated function to calculate metrics
    kpis = calculate_cumulative_metrics(df)
    
    # Get the all-time top 5 selling items from the exploded data
    top_items = exploded_df['item_name'].value_counts().head(5)
    top_items_list_str = "\n".join(
        [f"-> **{name}**: {count} sold" for name, count in top_items.items()]
    )

    # --- 2. Assemble the Report Content in Markdown Format ---
    report_content = f"""
# Cumulative Business Performance Report

This report summarizes the total business activity based on all available historical data from **{kpis['First Sale Date']}** to **{kpis['Last Sale Date']}**.

---

## 📈 All-Time Key Metrics

| Metric                      | Value                       |
| --------------------------- | --------------------------- |
| **Total Revenue** | {kpis['Total Revenue']}         |
| **Total Unique Receipts** | {kpis['Total Unique Receipts']}   |
| **Average Receipt Value** | {kpis['Average Receipt Value']}   |
| **Average Monthly Revenue** | {kpis['Average Monthly Revenue']} |

---

## 🍔 All-Time Top 5 Products

This list represents the most frequently sold individual items, including those from combo meals.

{top_items_list_str}

---

## 📊 Visualizations

See the accompanying plot images in this directory for more detailed trends:
- `cumulative_sales_trend.png`
"""

    # --- 3. Save the Report to a File ---
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "cumulative_summary_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip())
        
    logger.info(f"Cumulative summary report saved to: {report_path}")

    return report_path



# Function to orchestrate the cumulative report generation

def generate_cumulative_report(config):
    """
    Orchestrates the entire cumulative report generation process.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Cumulative Report Generation ---")

    # 1. Load all historical data from S3
    s3_bucket = config['s3_bucket']
    all_data_df = request_data(s3_bucket)

    if all_data_df.empty:
        logger.warning("No historical data found. Skipping cumulative report.")
        return

    # 2. Prepare the data for reporting
    cleaned_df = clean_data_for_reporting(all_data_df)
    exploded_df = explode_combo_items_advanced(cleaned_df)

    # 3. Define the output directory
    output_dir = config['project_dir'] / "reports" / "cumulative_report"
    
    # 4. Generate plots
    plot_cumulative_sales_trend(cleaned_df, output_dir)
    
    # 5. Generate the final .md summary file
    create_cumulative_summary_report(cleaned_df, exploded_df, output_dir)

    logger.info("--- Finished Cumulative Report Generation ---")



