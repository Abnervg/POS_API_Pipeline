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

def calculate_weekday_vs_weekend_performance(df):
    """
    Categorizes sales into Weekday vs. Weekend and calculates key metrics for each.

    Args:
        df (pd.DataFrame): The complete historical data (un-exploded for financial metrics).

    Returns:
        pd.DataFrame: A summary DataFrame with metrics for weekdays and weekends.
    """
    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # 1. Create a 'period_type' column (Weekday vs. Weekend)
    # Monday is 0, Sunday is 6. Weekends are Friday, Saturday, Sunday (4, 5, 6).
    df['period_type'] = df['shifted_time'].dt.dayofweek.apply(
        lambda x: 'Weekend' if x >= 4 else 'Weekday'
    )

    # 2. Group by the new category and calculate metrics
    performance_summary = df.groupby('period_type').agg(
        total_revenue=('price', 'sum'),
        total_receipts=('receipt_number', 'nunique')
    ).reset_index()

    # 3. Calculate the average sale value per receipt
    performance_summary['avg_sale_per_receipt'] = (
        performance_summary['total_revenue'] / performance_summary['total_receipts']
    )

    return performance_summary

def calculate_hourly_sales_traffic(df):
    """
    Calculates the number of unique receipts for each hour of each day of the week.

    Args:
        df (pd.DataFrame): The complete historical data.

    Returns:
        pd.DataFrame: A pivoted DataFrame ready for a heatmap, with days as rows
                      and hours as columns.
    """
    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # Create 'day_of_week' and 'hour' columns for grouping
    df['day_of_week'] = df['shifted_time'].dt.day_name()
    df['hour'] = df['shifted_time'].dt.hour

    # Group by day and hour, and count the number of unique receipts
    hourly_traffic = df.groupby(['day_of_week', 'hour'])['receipt_number'].nunique().reset_index()

    # Pivot the data to create a matrix: days on the y-axis, hours on the x-axis
    heatmap_data = hourly_traffic.pivot(index='day_of_week', columns='hour', values='receipt_number').fillna(0)

    # Order the days of the week correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    return heatmap_data


#-----Define plotting functions to visualize the data----

# Produce a bar chart comparing weekday vs. weekend performance
def plot_weekday_vs_weekend_comparison(df, output_dir):
    """
    Generates and saves bar charts comparing key metrics for weekdays vs. weekends.

    Args:
        df (pd.DataFrame): The complete historical data.
        output_dir (Path): The directory where the plot will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating weekday vs. weekend performance plot...")

    # 1. Get the prepared data
    performance_data = calculate_weekday_vs_weekend_performance(df)

    # 2. Create the plot with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Weekday vs. Weekend Performance Comparison', fontsize=20)

    # Plot 1: Total Revenue
    sns.barplot(ax=axes[0], data=performance_data, x='period_type', y='total_revenue', palette='viridis')
    axes[0].set_title('Total Revenue', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Total Sales ($)')
    axes[0].yaxis.set_major_formatter('${x:,.0f}')

    # Plot 2: Total Receipts
    sns.barplot(ax=axes[1], data=performance_data, x='period_type', y='total_receipts', palette='plasma')
    axes[1].set_title('Total Customer Traffic', fontsize=14)
    axes[1].set_xlabel('Period', fontsize=12)
    axes[1].set_ylabel('Number of Unique Receipts')

    # Plot 3: Average Sale per Receipt
    sns.barplot(ax=axes[2], data=performance_data, x='period_type', y='avg_sale_per_receipt', palette='magma')
    axes[2].set_title('Average Spend per Customer', fontsize=14)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Average Sale Value ($)')
    axes[2].yaxis.set_major_formatter('${x:,.2f}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 3. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "weekday_vs_weekend_performance.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Weekday vs. weekend performance plot saved to: {plot_path}")


# Produce a hourly sales heatmap
def plot_hourly_sales_heatmap(df, output_dir):
    """
    Generates and saves a heatmap of sales traffic by day and hour.

    Args:
        df (pd.DataFrame): The complete historical data.
        output_dir (Path): The directory where the plot will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating hourly sales heatmap...")

    # 1. Get the prepared data
    heatmap_data = calculate_hourly_sales_traffic(df)

    # 2. Create the heatmap plot
    plt.figure(figsize=(18, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,      # Show the number of receipts in each cell
        fmt=".0f",       # Format the numbers as integers
        cmap="YlGnBu",   # Use a sequential color map (Yellow-Green-Blue)
        linewidths=.5
    )

    # 3. Add titles and labels
    plt.title('Hourly Customer Traffic by Day of the Week', fontsize=18)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "hourly_sales_heatmap.png"
    plt.savefig(plot_path, bbox_inches='tight') # Use bbox_inches to prevent labels from being cut off
    plt.close()

    logger.info(f"Hourly sales heatmap saved to: {plot_path}")

# Produce a time series bar chart of total sales for each individual month
def plot_cumulative_sales_trend(df, output_dir):
    """
    Generates a time series bar chart of total sales for each individual month.

    Args:
        df (pd.DataFrame): The complete historical data.
        output_dir (Path): The directory where the plot will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly sales trend plot...")

    df = df.copy()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # --- Data Aggregation using groupby ---
    # Create a column with the year and month (e.g., '2025-07')
    df['month'] = df['shifted_time'].dt.strftime('%Y-%m')
    
    # Group by this new month column and sum the sales
    monthly_sales = df.groupby('month')['price'].sum().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=monthly_sales, 
        x='month', 
        y='price', 
        color='navy'
    )

    # Add titles and labels
    plt.title('Total Sales per Month', fontsize=18)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    
    # Format the y-axis to show dollar values
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "monthly_sales_trend.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Monthly sales trend plot saved to: {plot_path}")

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
    plot_hourly_sales_heatmap(cleaned_df, output_dir)
    plot_weekday_vs_weekend_comparison(cleaned_df, output_dir)
    
    # 5. Generate the final .md summary file
    create_cumulative_summary_report(cleaned_df, exploded_df, output_dir)

    logger.info("--- Finished Cumulative Report Generation ---")



