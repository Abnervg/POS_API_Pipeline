import logging
import pandas as pd
import re

# Data preparation functions for reporting
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
        else:
            return 'Refrescos y Aguas'

    beverages_df['category'] = beverages_df['item_name'].apply(assign_category)

    # 3. Get the count of each item within each category
    beverage_counts = beverages_df.groupby(['category', 'item_name']).size().reset_index(name='count')

    # 4. Calculate the percentage *within each category*
    total_per_category = beverage_counts.groupby('category')['count'].transform('sum')
    beverage_counts['percentage'] = (beverage_counts['count'] / total_per_category) * 100
    
    return beverage_counts
