 #This file produces a cumulative report of the data in the database.
# It aggregates data from various tables and generates a summary report.

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import re
from collections import Counter

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

def find_frequent_item_combinations(df):
    """
    Performs a market basket analysis to find which items are frequently
    purchased together on the same receipt.

    Args:
        df (pd.DataFrame): The cleaned historical data (BEFORE exploding combos).

    Returns:
        pd.DataFrame: A DataFrame showing the association rules.
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing market basket analysis...")

    # 1. Group items by receipt to create a "basket" for each transaction
    baskets = df.groupby('receipt_number')['item_name'].apply(list).tolist()

    # 2. Transform the data into the required format for the Apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)

    # 3. Find the frequent itemsets using the Apriori algorithm
    # min_support=0.01 means we're looking for combinations that appear in at least 1% of all transactions.
    frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

    if frequent_itemsets.empty:
        logger.warning("No frequent item combinations found with the current support threshold.")
        return pd.DataFrame()

    # 4. Generate the association rules
    # min_threshold=0.5 means we're looking for rules where the confidence is at least 50%.
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    
    # Sort by "lift" to find the most significant relationships
    rules.sort_values(by='lift', ascending=False, inplace=True)

    logger.info(f"Found {len(rules)} significant item combination rules.")
    return rules

def analyze_combo_choices(df):
    """
    Analyzes the choices made within combo items.

    Args:
        df (pd.DataFrame): The cleaned historical data (BEFORE exploding combos).

    Returns:
        dict: A dictionary where keys are combo names and values are the counts of each choice.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing choices within combo items...")

    combo_df = df[df['item_name'].str.contains('Combo', case=False, na=False)].copy()
    
    combo_analysis = {}

    for combo_name in combo_df['item_name'].unique():
        # Filter for just this specific combo
        specific_combo_df = combo_df[combo_df['item_name'] == combo_name]
        
        # This will hold all the choices made for this combo
        all_choices = []
        
        # Define the keys we care about (the choices)
        choice_keys = ['hamburguesa', 'refresco', 'papas', 'malteada']

        def extract_choices(modifier_string):
            if not isinstance(modifier_string, str): return []
            choices = []
            for part in modifier_string.split(';'):
                key = part.split('(')[0].lower()
                if any(item_key in key for item_key in choice_keys):
                    match = re.search(r'\((.*?)\)', part)
                    if match:
                        choices.append(match.group(1).strip())
            return choices

        # Apply the function to every modifier string for this combo
        list_of_choices = specific_combo_df['modifiers'].apply(extract_choices)
        
        # Flatten the list of lists into a single list
        for sublist in list_of_choices:
            all_choices.extend(sublist)
            
        # Count the occurrences of each choice
        combo_analysis[combo_name] = Counter(all_choices)

    return combo_analysis


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

def plot_combo_choices(df, output_dir):
    """
    Generates and saves a bar chart for each combo, showing the popularity of choices.

    Args:
        df (pd.DataFrame): The complete historical data (un-exploded).
        output_dir (Path): The directory where the plots will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating plots for combo choices...")

    # 1. Get the analysis data
    combo_analysis_data = analyze_combo_choices(df)

    # 2. Loop through each combo and create a plot for it
    for combo_name, choices_counter in combo_analysis_data.items():
        if not choices_counter:
            logger.warning(f"No choices found for combo '{combo_name}'. Skipping plot.")
            continue

        # Convert the Counter object to a DataFrame for easy plotting
        choices_df = pd.DataFrame(choices_counter.items(), columns=['Choice', 'Count']).sort_values(by='Count', ascending=False)

        # Create the plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=choices_df, x='Choice', y='Count', palette='rocket')

        # Add labels on top of the bars
        ax.bar_label(ax.containers[0])

        # Add titles and labels
        plt.title(f"Most Popular Choices for '{combo_name}'", fontsize=16)
        plt.xlabel('Choice', fontsize=12)
        plt.ylabel('Number of Times Chosen', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Sanitize the combo name to create a valid filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_]', '', combo_name.replace(' ', '_')).lower()
        
        # Save the plot
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"combo_choices_{safe_filename}.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Combo choice plot saved to: {plot_path}")

# Function to create a comprehensive cumulative summary report in Markdown format
def create_cumulative_summary_report(df, exploded_df, output_dir, association_rules_df):
    """
    Generates a comprehensive cumulative summary report in Markdown format.

    Args:
        df (pd.DataFrame): The complete historical data, cleaned but not exploded.
        exploded_df (pd.DataFrame): The historical data after exploding combo items.
        association_rules_df (pd.DataFrame): The results from the market basket analysis.
        output_dir (Path): The directory to save the report file.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive cumulative summary report...")

    # --- 1. Calculate Key Performance Indicators (KPIs) ---
    kpis = calculate_cumulative_metrics(df)
    
    top_items = exploded_df['item_name'].value_counts().head(5)
    top_items_list_str = "\n".join(
        [f"-> **{name}**: {count} sold" for name, count in top_items.items()]
    )

    # --- 2. Format the Market Basket Analysis Results ---
    market_basket_table = "| Items Purchased Together | Likelihood |\n| :--- | :--- |\n"
    # Take the top 5 most confident rules
    for _, row in association_rules_df.head(5).iterrows():
        # Convert frozensets to readable strings
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        confidence = row['confidence']
        market_basket_table += f"| If a customer buys **{antecedents}**, they also buy **{consequents}** | {confidence:.0%} |\n"


    # --- 3. Assemble the Report Content in Markdown Format ---
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

## 🛒 Popular Product Combinations

This table shows which items are frequently purchased together, based on a confidence threshold of 50%.

{market_basket_table}

---

## 📊 Visualizations

See the accompanying plot images in this directory for more detailed trends:
- `cumulative_sales_trend.png`
"""

    # --- 4. Save the Report to a File ---
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
    plot_combo_choices(cleaned_df, output_dir)
    
    # 5. Generate the final .md summary file
    association_rules_df = find_frequent_item_combinations(exploded_df)
    create_cumulative_summary_report(cleaned_df, exploded_df, output_dir, association_rules_df)

    logger.info("--- Finished Cumulative Report Generation ---")



