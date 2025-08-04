 #This file produces a cumulative report of the data in the database.
# It aggregates data from various tables and generates a summary report.

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging



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
    Loads all partitioned monthly data from S3 and merges them into a single dataframe for cumulative analysis

        Args:bucketname(str): S3 bucketname where data is stored

        Returns:  
    """
    logger = logging.getLogger(__name__)
    # Define S3 path
    s3_path = f"s3://{bucket_name}/curated_data/**/*.parquet"

    logger.info(f"Loading historical data from S3 bucket {bucket_name}")
    # Merge data
    try:
        historical_df = pd.read_parquet(s3_path)
        logger.info(f"Successfully extracted {len(historical_df)} records")
        return historical_df
    except FileNotFoundError:
        logger.warning(f"Failed to find data in S3 {bucket_name}, returning empty DataFrame")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found
    except Exception as e:
        logger.error(f"An error occurred while loading data from S3: {e}")
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
        [f"1. **{name}**: {count} sold" for name, count in top_items.items()]
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
    Generates a cumulative report by aggregating data from the S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket containing the data.

        Returns:
            pd.DataFrame: A DataFrame containing the cumulative report.
    """
    logger = logging.getLogger(__name__)
    
    # Define output directory for plots
    output_dir = config['project_dir'] / "reports" / "cumulative"

    # Load historical data
    s3_bucket = config['bucket_name']
    all_data = request_data(s3_bucket)

    logger.info(f"--- Starting Cumulative Report Generation ---")
    if all_data.empty:
        logger.info("No historical data found. Returning empty report.")
        return pd.DataFrame()
    
    # This section performs data transformations
    cumulative_kpis = calculate_cumulative_metrics(all_data)

    # This section produce different plots to use for the report
    plot_cumulative_sales_trend(all_data, output_dir)

    # This section generates the final report
    
    #

    logger.info(f"--- Finished Cumulative Report Generation ---")



