import pandas as pd
import logging
import matplotlib.pyplot as plt
import awswrangler as wr
from pathlib import Path
import numpy as np
import seaborn as sns
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from .utils import convert_md_to_pdf, send_report_by_email
import boto3


# Import your data preparation functions
from .data_preparation import clean_data_for_reporting, explode_combo_items_advanced
from .data_preparation import calculate_beverage_distribution, calculate_mayo_percentages_and_counts, calculate_sales_by_day_of_week
from .data_preparation import calculate_daily_sales_metrics, calculate_sales_by_day_for_comparison, calculate_mayo_distribution_by_month, calculate_beverage_distribution_by_month
from .data_preparation import get_top_products, calculate_daily_sales_for_comparison
# Requests monthly data

def request_monthly_data(year: int, month: int, database_name: str, table_name: str) -> pd.DataFrame:
    """
    Queries Athena to get a clean, deduplicated DataFrame for a specific
    month and the month prior to it, correctly handling zero-padded month strings.

    Args:
        year (int): The year of the primary report month.
        month (int): The primary report month.
        database_name (str): The name of the Athena database.
        table_name (str): The name of the Athena table.

    Returns:
        pd.DataFrame: A single DataFrame containing two months of clean data.
    """
    logger = logging.getLogger(__name__)

    # --- KEY CHANGE ---
    # Calculate date ranges and immediately format them into the strings
    # that match your Athena table's format.
    report_month_date = datetime(year, month-1, 1)
    previous_month_date = report_month_date - relativedelta(months=1)

    # Use strftime to get zero-padded month ('%m') and year ('%Y') strings
    report_year_str = report_month_date.strftime('%Y')
    report_month_str = report_month_date.strftime('%m') # e.g., 9 -> '09'
    previous_year_str = previous_month_date.strftime('%Y')
    previous_month_str = previous_month_date.strftime('%m') # e.g., 8 -> '08'
    
    # This SQL query will be executed by Athena.
    # It now uses the pre-formatted string variables to ensure the WHERE clause is correct.
    query = f"""
        WITH latest_records AS (
            SELECT 
                *,
                ROW_NUMBER() OVER(
                    PARTITION BY receipt_number, item_name 
                    ORDER BY shifted_time DESC
                ) as rank_num
            FROM 
                "{table_name}"
            WHERE 
                (year = '{report_year_str}' AND month = '{report_month_str}') OR 
                (year = '{previous_year_str}' AND month = '{previous_month_str}')
        )
        SELECT 
            receipt_number, datetime, order_type, item_name, cost, price, 
            total_money, modifiers, payment_type, shifted_time, time_slot
        FROM 
            latest_records 
        WHERE 
            rank_num = 1
    """

    logger.info(f"Executing Athena query on table '{table_name}' for months {previous_year_str}-{previous_month_str} and {report_year_str}-{report_month_str}")

    try:
        # Set up a boto3 session (ensure AWS credentials are configured)
        sess = boto3.Session(region_name="us-east-1")
        
        # Use awswrangler to run the query and get a pandas DataFrame
        combined_df = wr.athena.read_sql_query(
            sql=query,
            database=database_name,
            boto3_session=sess
        )
        logger.info(f"Successfully loaded a total of {len(combined_df)} records.")
        return combined_df

    except Exception as e:
        logger.error(f"An Athena query error occurred: {e}")
        return pd.DataFrame()


# --- Plotting Functions ---

def plot_monthly_mayo_comparison(df, output_dir):
    """
    Generates a grouped bar chart comparing mayonnaise preferences between months.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly comparison plot for mayonnaise preference...")

    # 1. Get the prepared data
    comparison_data = calculate_mayo_distribution_by_month(df)

    # 2. Create the plot using Seaborn's catplot for easy faceting
    g = sns.catplot(
        data=comparison_data,
        x='item_name',
        y='count',
        hue='mayo_type',
        col='month',  # This creates a separate subplot for each month
        kind='bar',
        palette='tab10',
        height=6,
        aspect=1.2
    )

    # 3. Add titles and labels
    g.fig.suptitle('Monthly Comparison of Mayonnaise Preference per Burger', y=1.03, fontsize=16)
    g.set_axis_labels("Burger Type", "Number of Items Sold")
    g.set_titles("Month: {col_name}")
    g.legend.set_title("Mayo Type")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "monthly_mayo_preference_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Monthly mayo comparison plot saved to: {plot_path}")

def plot_monthly_beverage_comparison(df, output_dir):
    """
    Creates a faceted bar chart to compare beverage distribution between months,
    ensuring correct alignment.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly comparison plot for beverage distribution...")

    # 1. Get the prepared data
    comparison_data = calculate_beverage_distribution_by_month(df)

    # 2. Create the plot using Seaborn's catplot
    g = sns.catplot(
        data=comparison_data,
        x='category',
        y='count',
        hue='item_name',
        col='month',
        kind='bar',
        palette='tab10',
        height=7,
        aspect=1.1,
        order=['Aguas', 'Malteadas', 'Refrescos'] # This ensures alignment
    )

    # 3. Add titles and labels
    g.fig.suptitle('Monthly Comparison of Beverage Sales', y=1.03, fontsize=18)
    g.set_axis_labels("Beverage Category", "Number of Items Sold")
    g.set_titles("Month: {col_name}")
    
    # --- FIX: Move the automatically generated legend outside the plot area ---
    g.legend.set_title("Beverage Type")
    g.legend.set_bbox_to_anchor((1.02, 0.5)) # Move legend to the right
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])

    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "monthly_beverage_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Monthly beverage comparison plot saved to: {plot_path}")

def create_top_products_plot(df, output_dir):
    """
    Generates a bar chart of the top 5 sold products for a given month
    and saves it to a file.

    Args:
        df (pd.DataFrame): The DataFrame for a single month (already exploded).
        output_dir (Path): The directory where the plot image will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating top products plot...")
    
    # 1. Get the top 5 items from the data
    top_products_df = get_top_products(df, top_n=5)
    
    # 2. Create the plot
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        data=top_products_df, 
        x='item_name', 
        y='items_sold', 
        hue='item_name', # Use hue to assign different colors
        palette='viridis', 
        legend=False
    )
    
    # 3. Add labels on top of the bars
    ax.bar_label(ax.containers[0])
    
    # 4. Add titles and labels for clarity
    plt.title('Top 5 Most Sold Items This Month', fontsize=16)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 5. Save the plot to a file
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "top_5_products.png"
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory
    
    logger.info(f"Top products plot saved to: {plot_path}")




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

    # 3. Create the plot
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.6, colormap='tab10')

    # 4. Add percentage labels
    for container in ax.containers:
        item_name = container.get_label()
        labels = []
        for i, bar in enumerate(container):
            category_name = counts_pivot.index[i]
            try:
                percentage = percentages_pivot.loc[category_name, item_name]
                if percentage > 5:
                    labels.append(f'{percentage:.0f}%')
                else:
                    labels.append('')
            except KeyError:
                labels.append('')
        ax.bar_label(container, labels=labels, label_type='center', color='white', weight='bold')

    # 5. Create Custom Grouped Legends (this logic is dynamic)
    handles, labels = ax.get_legend_handles_labels()
    item_to_category = pd.Series(data_for_plot.category.values, index=data_for_plot.item_name).to_dict()
    
    category_legends = {cat: [] for cat in data_for_plot['category'].unique()}
    for handle, label in zip(handles, labels):
        category = item_to_category.get(label)
        if category:
            category_legends[category].append((handle, label))
            
    if ax.get_legend():
        ax.get_legend().remove()

    legend_y_start = 1.02
    for category, items in category_legends.items():
        if not items: continue
        cat_handles, cat_labels = zip(*items)
        legend = ax.legend(cat_handles, cat_labels, title=f'-- {category} --', 
                           bbox_to_anchor=(1.02, legend_y_start), 
                           loc='upper left', 
                           title_fontsize='12',
                           fontsize='10')
        ax.add_artist(legend)
        legend_y_start -= 0.3

    # 6. Add titles and labels
    plt.title('Beverage Sales Distribution', fontsize=16)
    plt.xlabel('Beverage Category', fontsize=12)
    plt.ylabel('Number of Items Sold', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])

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

def plot_daily_sales_comparison(df, output_dir):
    """
    Generates a comparative line plot of daily customer traffic for two months.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating daily sales comparison plot...")

    # 1. Get the daily metrics, prepared for comparison
    daily_data = calculate_daily_sales_for_comparison(df)

    # 2. Create the plot with a single Y-axis
    plt.figure(figsize=(15, 8))
    
    # Use 'hue' to create a separate line for each month
    ax = sns.lineplot(
        data=daily_data, 
        x='day_of_month', 
        y='unique_receipts', 
        hue='month',
        palette=['gray', 'black'], 
        marker='o'
    )
    
    # 3. Add titles and formatting
    ax.set_xlabel('Day of the Month', fontsize=12)
    ax.set_ylabel('Number of Unique Receipts', fontsize=12)
    plt.title('Daily Customer Traffic: Month-over-Month Comparison', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Month')
    plt.tight_layout()
    
    # 4. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "daily_sales_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Daily sales comparison plot saved to: {plot_path}")


def plot_monthly_comparison_by_weekday(df, output_dir):
    """
    Generates a comparative line plot showing sales traffic by day of the week
    for two different months.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly comparison plot for sales by weekday...")

    # 1. Get the prepared data
    comparison_data = calculate_sales_by_day_for_comparison(df)

    # 2. Filter out the 'Otro' category before plotting
    comparison_data = comparison_data[comparison_data['order_category'] != 'Otro']

    # 3. Create the plot
    plt.figure(figsize=(14, 8))
    
    # Use hue for the month and style for the order category
    ax = sns.lineplot(
        data=comparison_data,
        x='day_of_week',
        y='count',
        hue='month',          # Separates lines by month
        style='order_category', # Creates different dashing for order types
        palette=['red', 'blue'], # Sets the colors for each month
        markers=True,
        markersize=10,
        dashes=True,
        linewidth=2
    )

    # 4. Add titles and labels
    plt.title('Monthly Comparison of Sales Traffic by Day', fontsize=18)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Unique Receipts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Month & Order Type')
    plt.tight_layout()

    # 5. Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "monthly_comparison_by_weekday.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Monthly comparison plot saved to: {plot_path}")


# Monthly report template

def create_monthly_summary_report(df, output_dir):
    """
    Generates a comprehensive summary report in Markdown format for the last 
    complete month, with detailed comparisons to the month prior.

    Args:
        df (pd.DataFrame): DataFrame containing data for the last two months.
        output_dir (Path): The directory to save the report file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating monthly summary report...")

    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'])
    df['month'] = df['shifted_time'].dt.strftime('%Y-%m')

    # --- 1. Separate Data and Calculate KPIs ---
    now = datetime.now()
    report_month_date = now - relativedelta(months=1)
    comparison_month_date = now - relativedelta(months=2)

    report_month_tag = report_month_date.strftime('%Y-%m')
    comparison_month_tag = comparison_month_date.strftime('%Y-%m')

    report_month_df = df[df['month'] == report_month_tag]
    comparison_month_df = df[df['month'] == comparison_month_tag]

    def calculate_kpis(data):
        if data.empty: return {'revenue': 0, 'receipts': 0}
        return {'revenue': data['price'].sum(), 'receipts': data['receipt_number'].nunique()}

    kpis_report = calculate_kpis(report_month_df)
    kpis_comparison = calculate_kpis(comparison_month_df)

    def pct_change(current, previous):
        if previous == 0: return " (new)"
        change = ((current - previous) / previous) * 100
        return f" ({change:+.1f}%)"

    revenue_change = pct_change(kpis_report['revenue'], kpis_comparison['revenue'])
    receipts_change = pct_change(kpis_report['receipts'], kpis_comparison['receipts'])

    # --- 2. Calculate Top 5 Products for Both Months ---
    exploded_report_df = explode_combo_items_advanced(report_month_df)
    exploded_comparison_df = explode_combo_items_advanced(comparison_month_df)
    
    top_5_report = get_top_products(exploded_report_df, 5)
    top_5_comparison = get_top_products(exploded_comparison_df, 5)

    # --- 3. Assemble the Report Content ---
    report_content = f"""
# Monthly Sales Report: {report_month_tag}

This report analyzes sales performance for **{report_month_tag}** and compares it to **{comparison_month_tag}**.

---

## 📈 Monthly Performance Summary

| Metric                      | Report Month ({report_month_tag}) | Comparison Month ({comparison_month_tag}) |
| --------------------------- | --------------------------- | ----------------------------- |
| **Total Revenue** | `${kpis_report['revenue']:,.2f}` **{revenue_change}** | `${kpis_comparison['revenue']:,.2f}`       |
| **Total Unique Receipts** | `{kpis_report['receipts']:,}` **{receipts_change}** | `{kpis_comparison['receipts']:,}`       |

---

## 🍔 Top 5 Products Comparison

| Rank | Top Products ({report_month_tag}) | Items Sold | | Top Products ({comparison_month_tag}) | Items Sold |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | {top_5_report.iloc[0]['item_name']} | {top_5_report.iloc[0]['items_sold']} | | {top_5_comparison.iloc[0]['item_name']} | {top_5_comparison.iloc[0]['items_sold']} |
| **2** | {top_5_report.iloc[1]['item_name']} | {top_5_report.iloc[1]['items_sold']} | | {top_5_comparison.iloc[1]['item_name']} | {top_5_comparison.iloc[1]['items_sold']} |
| **3** | {top_5_report.iloc[2]['item_name']} | {top_5_report.iloc[2]['items_sold']} | | {top_5_comparison.iloc[2]['item_name']} | {top_5_comparison.iloc[2]['items_sold']} |
| **4** | {top_5_report.iloc[3]['item_name']} | {top_5_report.iloc[3]['items_sold']} | | {top_5_comparison.iloc[3]['item_name']} | {top_5_comparison.iloc[3]['items_sold']} |
| **5** | {top_5_report.iloc[4]['item_name']} | {top_5_report.iloc[4]['items_sold']} | | {top_5_comparison.iloc[4]['item_name']} | {top_5_comparison.iloc[4]['items_sold']} |

---

## 📊 Visual Comparisons

### Daily Customer Traffic

![Daily Sales Comparison](./daily_sales_comparison.png)

***Discussion:*** [This plot shows the daily customer traffic for both months.]

### Sales Traffic by Day of the Week

![Monthly Comparison by Weekday](./monthly_comparison_by_weekday.png)

***Discussion:*** [This plot shows the sales traffic by day of the week for both months.]

### Mayonnaise Preference per Burger

![Monthly Mayo Comparison](./monthly_mayo_preference_comparison.png)

***Discussion:*** [This plot shows the preference for mayonnaise on burgers for both months.]

### Beverage Sales Distribution

![Monthly Beverage Comparison](./monthly_beverage_comparison.png)

***Discussion:*** [This plot shows the beverage sales distribution for both months.]
"""

    # --- 4. Save the Report ---
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"monthly_summary_{report_month_tag}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip())
        
    logger.info(f"Monthly summary report saved to: {report_path}")
    return report_path




# --- Main Function to Generate Monthly Report ---

def generate_monthly_report(config):
    """
    Orchestrates the entire monthly report generation and delivery process.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Monthly Report Generation ---")

    # --- 1. Load Data ---
    athena_database = config['athena_database']
    athena_table = config['athena_table']

    current_month = datetime.now().month
    current_year = datetime.now().year

    two_months_df = request_monthly_data(current_year, current_month, athena_database, athena_table)

    if two_months_df.empty:
        logger.warning("No data found for the last two months. Skipping monthly report.")
        return

    # --- 2. Prepare Data ---
    # This step creates the two versions of the data we need for different analyses
    cleaned_df = clean_data_for_reporting(two_months_df)
    exploded_df = explode_combo_items_advanced(cleaned_df)

    # --- 3. Define Output Paths and File Tags ---
    report_month = datetime.now() - relativedelta(months=1)
    file_tag = report_month.strftime('%Y-%m')
    frequency = 'monthly'  # This is used for the report name and email subject
    report_output_dir = config['project_dir'] / "reports" / f"monthly_report_{file_tag}"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Generate All Report Artifacts ---
    logger.info("Generating plots...")
    # These plots compare months, so they use the full two-month DataFrame
    plot_monthly_beverage_comparison(cleaned_df, report_output_dir)
    plot_monthly_mayo_comparison(cleaned_df, report_output_dir)
    plot_monthly_comparison_by_weekday(cleaned_df, report_output_dir)
    plot_daily_sales_comparison(cleaned_df, report_output_dir)
    
    # This plot is for the most recent month's top products
    report_month_df = exploded_df[exploded_df['shifted_time'].dt.strftime('%Y-%m') == file_tag]
    create_top_products_plot(report_month_df, report_output_dir)

    logger.info("Generating summary document...")
    report_md_path = create_monthly_summary_report(cleaned_df, report_output_dir)
    
    # --- 5. Deliver the Final Report ---
    if report_md_path:
        logger.info("Converting report to PDF...")
        pdf_path = convert_md_to_pdf(report_md_path, report_output_dir)
        if pdf_path:
            logger.info("Sending report by email...")
            # Get recipient from config for better practice
            recipient_email = config.get("recipient_email", "default.recipient@example.com")
            send_report_by_email(pdf_path, recipient_email, file_tag, frequency)
        

    logger.info(f"--- Monthly Report process completed. Artifacts are in: {report_output_dir} ---")
