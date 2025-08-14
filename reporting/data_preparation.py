import logging
import pandas as pd
import re



# Data preparation functions for reporting

def get_top_products(df, top_n=5):
    """
    Counts how many times each item appears in the DataFrame and returns the top N.
    This function should be used with the 'exploded' DataFrame for accurate counts.

    Args:
        df (pd.DataFrame): The DataFrame containing item sales data.
        top_n (int): The number of top products to return.

    Returns:
        pd.DataFrame: A DataFrame with the top N products and their sales counts.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating top {top_n} sold products...")

    # .value_counts() is the most efficient way to count and sort items
    top_items_series = df['item_name'].value_counts().head(top_n)
    
    # Convert the resulting Series to a DataFrame
    top_items_df = top_items_series.reset_index()
    
    # Rename the columns for clarity
    top_items_df.columns = ['item_name', 'items_sold']
    
    return top_items_df

def calculate_sales_by_day_for_comparison(df):
    """
    Prepares data for a two-month comparison by grouping sales by month,
    day of the week, and order category.
    """
    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)
    
    # Create necessary columns for grouping
    df['month'] = df['shifted_time'].dt.strftime('%Y-%m')
    df['day_of_week'] = df['shifted_time'].dt.day_name()
    
    # Convert 'day_of_week' to an ordered Categorical type to ensure correct sorting
    day_order = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

    # Create Order Type Categories
    def assign_order_category(order_type):
        if not isinstance(order_type, str): return 'Otro'
        if 'mesa' in order_type.lower(): return 'Restaurante'
        if 'domicilio' in order_type.lower(): return 'A domicilio'
        if 'llevar' in order_type.lower(): return 'Para llevar'
        return 'Otro'

    df['order_category'] = df['order_type'].apply(assign_order_category)

    # DEBUGGING: Print unique order types and their categories
    # Filter for the 'Otro' category and find the unique original values
    #otro_values = df[df['order_category'] == 'Otro']['order_type'].unique()

    # Print the results
    #print("The following order_type values are being categorized as 'Otro':")
    #print(otro_values)
    
    # Group by all three columns and count unique receipts
    categorized_sales = df.groupby(['month', 'day_of_week', 'order_category'], observed=False)['receipt_number'].nunique().reset_index()
    categorized_sales.rename(columns={'receipt_number': 'count'}, inplace=True)
    
    return categorized_sales

def clean_data_for_reporting(df):
    """
    Cleans the DataFrame for reporting purposes.
    
    Parameters:
    df (DataFrame): The DataFrame to clean.
    
    Returns:
    DataFrame: The cleaned DataFrame.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Data Cleaning for Reporting ---")
    df = df.copy()

    # 1. Correct Data Types
    # Ensure numeric columns are numbers, coercing errors to NaN (Not a Number)
    numeric_cols = ['cost', 'price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure datetime is a proper datetime object
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # 2. Handle Missing Values
    # Drop rows where key information is missing
    df.dropna(subset=['datetime', 'receipt_number'], inplace=True)
    
    # Fill missing numeric values with 0 and text with 'Unknown'
    df['price'] = df['price'].fillna(0)
    df['cost'] = df['cost'].fillna(0)
    df['item_name'] = df['item_name'].fillna('Unknown')

    # 3. Feature Engineering: Create new columns for analysis
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour_of_day'] = df['datetime'].dt.hour
    logger.info("Data cleaning and feature engineering complete.")
    return df


# Explode combo items into individual rows with modifiers
def explode_combo_items_advanced(df):
    """
    Finds combo items, parses their modifiers, and creates new, clean rows for
    each individual item found within the combo, correctly associating sub-modifiers.
    *Note*: This function removes the price and cost from individual items, as they belong to the combo.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting advanced explosion of combo items...")

    df = df.copy()

    # 1. Isolate combo rows from regular items
    is_combo = df['item_name'].str.contains('Combo', case=False, na=False)
    combo_df = df[is_combo].copy()
    non_combo_df = df[~is_combo]

    if combo_df.empty:
        logger.info("No combo items found to explode.")
        return df

    # This list will hold all the new rows created from the combos
    new_rows = []

    # 2. Iterate through each combo row to perform detailed parsing
    for index, combo_row in combo_df.iterrows():
        modifier_string = combo_row.get('modifiers', '')
        if not isinstance(modifier_string, str):
            continue

        # --- Advanced Parsing Logic ---
        all_modifiers = modifier_string.split(';')
        
        # Separate main items (like Hamburguesa, Refresco) from sub-modifiers (like Mayonesa)
        main_items_raw = [m for m in all_modifiers if 'hamburguesa' in m.lower() or 'refresco' in m.lower()]
        sub_mods_raw = [m for m in all_modifiers if 'mayonesa' in m.lower()]

        # Process each main item found in the combo
        for i, item_mod in enumerate(main_items_raw):
            new_row = combo_row.to_dict() # Start with a copy of the original combo row's data
            
            # Extract the specific item name from inside the parentheses
            item_name_match = re.search(r'\((.*?)\)', item_mod)
            if not item_name_match:
                continue
            
            raw_item_name = item_name_match.group(1).strip()

            # Apply naming rules
            if 'smash' in raw_item_name.lower():
                new_item_name = 'Smash Burger'
            elif 'chiken' in raw_item_name.lower(): # Correcting potential typo
                new_item_name = 'Chicken Burger'
            elif 'Coca' in raw_item_name:
                new_item_name = 'Coca Cola'
            else: # For things like 'Agua Natural' from 'Refresco Sabor(Agua Natural)'
                new_item_name = raw_item_name

            # Associate the correct sub-modifier (e.g., Mayonesa)
            # This assumes the first Mayonesa goes with the first Hamburguesa, etc.
            associated_modifier = None
            if 'hamburguesa' in item_mod.lower() and i < len(sub_mods_raw):
                associated_modifier = sub_mods_raw[i]

            # Update the new row with the correct, un-nested data
            new_row['item_name'] = new_item_name
            new_row['modifiers'] = associated_modifier
            new_row['price'] = 0  # The price belongs to the combo, not the individual item
            new_row['cost'] = 0

            new_rows.append(new_row)

    # 3. Create a DataFrame from the newly created rows
    if not new_rows:
        logger.warning("No new item rows were generated from combos.")
        return non_combo_df # Return only the non-combo items if parsing failed

    exploded_df = pd.DataFrame(new_rows)

    # 4. Combine the new item rows with the original non-combo rows
    final_df = pd.concat([non_combo_df, exploded_df], ignore_index=True)
    
    logger.info(f"Original row count: {len(df)}, New row count after advanced exploding: {len(final_df)}")
    return final_df

#Helper function to plot mayonnaise preferences
def calculate_mayo_distribution_by_month(df):
    """
    Calculates the count of each mayonnaise type for each burger, grouped by month.
    """
    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)

    # 1. Filter for burgers and then for Mayonesa modifiers
    all_burgers = df[df['item_name'].str.contains("Burger|Smash", case=False, na=False)]
    mayo_burgers = all_burgers[all_burgers['modifiers'].str.contains("Mayonesa", case=False, na=False)].copy()

    # 2. Extract the specific mayo type
    mayo_burgers['mayo_type'] = mayo_burgers['modifiers'].str.extract(r'Mayonesa\((.*?)\)')
    
    # 3. Standardize the mayo types
    def standardize_mayo(mayo_name):
        if isinstance(mayo_name, str) and "sin mayonesa" in mayo_name.lower():
            return "Natural"
        return mayo_name
        
    mayo_burgers['mayo_type'] = mayo_burgers['mayo_type'].apply(standardize_mayo)

    # 4. Create a 'month' column for grouping
    mayo_burgers['month'] = mayo_burgers['shifted_time'].dt.strftime('%Y-%m')

    # 5. Group by month, burger name, and mayo type to get the counts
    mayo_counts = mayo_burgers.groupby(['month', 'item_name', 'mayo_type']).size().reset_index(name='count')
    
    return mayo_counts

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

    def standardize_mayotypes(df):
        for mod in df["mayo_type"]:
            lowed = mod.lower()
            if lowed.str.contains(
                "sin mayonesa"):
                return "Natural"
    mayo_burgers.apply(standardize_mayotypes)

    # Get the count of each mayo type per burger

    mayo_counts = mayo_burgers.groupby(['item_name', 'mayo_type']).size().reset_index(name='count')

    # Get the total number of mayo burgers for each type to calculate percentages

    total_burgers_per_item = mayo_counts.groupby('item_name')['count'].transform('sum')

    mayo_counts['percentage'] = (mayo_counts['count'] / total_burgers_per_item) * 100

    return mayo_counts

#Helper function to plot beverages sold

def calculate_beverage_distribution(df):
    """
    Categorizes beverages, then calculates the raw count and percentage
    distribution within each category.

    Returns:
        pd.DataFrame: A DataFrame with columns ['category', 'item_name', 'count', 'percentage']
    """
    # 1. Filter for all beverage items
    beverage_keywords = "Refresco|Malteada|Coca|Squirt|Agua|Manzanita"
    beverages_df = df[df['item_name'].str.contains(beverage_keywords, case=False, na=False)].copy()

    # 2. Create a new 'category' column
    def assign_category(item_name):
        if 'malteada' in item_name.lower():
            return 'Malteadas'
        elif 'agua' in item_name.lower():
            return 'Aguas'
        else:
            return 'Refrescos'
    
    beverages_df['category'] = beverages_df['item_name'].apply(assign_category)

    # 3. Get the count of each item within each category
    beverage_counts = beverages_df.groupby(['category', 'item_name']).size().reset_index(name='count')

    # 4. Calculate the percentage *within each category*
    total_per_category = beverage_counts.groupby('category')['count'].transform('sum')
    beverage_counts['percentage'] = (beverage_counts['count'] / total_per_category) * 100
    
    return beverage_counts

def calculate_beverage_distribution_by_month(df):
    """
    Categorizes beverages and calculates the raw count for each item,
    grouped by month.
    """
    # 1. Filter for all beverage items
    beverage_keywords = "Refresco|Malteada|Coca|Squirt|Agua|Manzanita"
    beverages_df = df[df['item_name'].str.contains(beverage_keywords, case=False, na=False)].copy()

    # --- Standardization Step for Water ---
    def standardize_beverage_names(name):
        if not isinstance(name, str): return name
        name_lower = name.lower()
        if 'mineral' in name_lower:
            return 'Agua Mineral'
        if 'natural' in name_lower or 'embotellada' in name_lower:
            return 'Agua Embotellada'
        return name
    beverages_df['item_name'] = beverages_df['item_name'].apply(standardize_beverage_names)
    
    # 2. Create 'month' and 'category' columns
    beverages_df['month'] = pd.to_datetime(beverages_df['shifted_time']).dt.strftime('%Y-%m')
    def assign_category(item_name):
        if 'malteada' in item_name.lower(): return 'Malteadas'
        elif 'agua' in item_name.lower(): return 'Aguas'
        else: return 'Refrescos'
    beverages_df['category'] = beverages_df['item_name'].apply(assign_category)

    # 3. Get the counts for existing data
    beverage_counts = beverages_df.groupby(['month', 'category', 'item_name']).size().reset_index(name='count')
    
    return beverage_counts

#Calculate sells by day of week
def calculate_sales_by_day_of_week(df):
    """
    Calculates sales traffic by day of the week, both total and by order type.
    """
    df = df.copy()
    # Ensure 'shifted_time' column exists and is the correct type
    if 'shifted_time' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['shifted_time']):
        df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
        df.dropna(subset=['shifted_time'], inplace=True)
        
    df['day_of_week'] = df['shifted_time'].dt.day_name()
    
    # --- FIX: Convert 'day_of_week' to an ordered Categorical type ---
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

    # --- Create Order Type Categories ---
    def assign_order_category(order_type):
        if not isinstance(order_type, str):
            return 'Otro'
        if 'Mesa' in order_type:
            return 'Restaurante'
        if 'domicilio' in order_type.lower():
            return 'A domicilio'
        if 'llevar' in order_type.lower():
            return 'Para llevar'
        return 'Otro'

    df['order_category'] = df['order_type'].apply(assign_order_category)
    
    # --- Calculate total sales by day ---
    # The groupby will now respect the categorical order, and we use observed=False
    # to ensure all days of the week are present, even if they have 0 sales.
    total_sales_by_day = df.groupby('day_of_week', observed=False)['receipt_number'].nunique()
    
    # --- Calculate sales by day AND category ---
    categorized_sales = df.groupby(['day_of_week', 'order_category'], observed=False)['receipt_number'].nunique().reset_index()
    categorized_sales.rename(columns={'receipt_number': 'count'}, inplace=True)
    
    return total_sales_by_day, categorized_sales

def calculate_daily_sales_metrics(df):
    """
    Calculates total sales and unique receipts for each day of the month.
    This function should be used with data BEFORE exploding combo items.
    """
    df = df.copy()
    # Ensure 'shifted_time' is a datetime object
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)
    
    # Group by date and aggregate
    daily_metrics = df.groupby(df['shifted_time'].dt.date).agg(
        total_sales=('price', 'sum'),
        unique_receipts=('receipt_number', 'nunique')
    ).reset_index()
    
    # Rename the date column for clarity
    daily_metrics.rename(columns={'shifted_time': 'date'}, inplace=True)
    
    return daily_metrics

def calculate_daily_sales_for_comparison(df):
    """
    Calculates total sales and unique receipts for each day, keeping the
    month separate for comparison.
    """
    df = df.copy()
    df['shifted_time'] = pd.to_datetime(df['shifted_time'], errors='coerce')
    df.dropna(subset=['shifted_time'], inplace=True)
    
    # Create 'month' and 'day' columns for grouping
    df['month'] = df['shifted_time'].dt.strftime('%Y-%m')
    df['day_of_month'] = df['shifted_time'].dt.day
    
    # Group by both month and day and aggregate
    daily_metrics = df.groupby(['month', 'day_of_month']).agg(
        total_sales=('price', 'sum'),
        unique_receipts=('receipt_number', 'nunique')
    ).reset_index()
    
    return daily_metrics