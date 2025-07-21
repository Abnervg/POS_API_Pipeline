import logging
import pandas as pd

def cleaning_data_for_reporting(df):
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
    df['price'].fillna(0, inplace=True)
    df['cost'].fillna(0, inplace=True)
    df['item_name'].fillna('Unknown', inplace=True)

    # 3. Feature Engineering: Create new columns for analysis
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour_of_day'] = df['datetime'].dt.hour
    
    logger.info("Data cleaning and feature engineering complete.")
    return df

def explode_combo_items(df):
    """
    Finds combo items, parses their modifiers, and creates new rows for each
    individual item found within the combo.
    """
    logger = logging.getLogger(__name__)
    logger.info("Exploding combo items into individual rows...")

    df = df.copy()

    # Isolate combo rows from regular items
    is_combo = df['item_name'].str.contains('Combo', case=False, na=False)
    combo_df = df[is_combo]
    non_combo_df = df[~is_combo]

    if combo_df.empty:
        logger.info("No combo items found to explode.")
        return df

    def extract_items(modifier_string):
        """Helper function to parse the modifier string for items."""
        if not isinstance(modifier_string, str):
            return []
        
        # This list defines which modifier keys we consider to be main items.
        # You may need to adjust this based on your data.
        item_keys = ['hamburguesa', 'refresco', 'papas', 'malteada']
        
        items = []
        for part in modifier_string.split(';'):
            key = part.split('(')[0].lower()
            if any(item_key in key for item_key in item_keys):
                match = re.search(r'\((.*?)\)', part)
                if match:
                    items.append(match.group(1).strip())
        return items

    # Create a new column with the list of extracted items
    combo_df['parsed_items'] = combo_df['modifiers'].apply(extract_items)

    # Explode the DataFrame, creating a new row for each item
    exploded_df = combo_df.explode('parsed_items').dropna(subset=['parsed_items'])

    # Update the new rows with the correct item name
    exploded_df['item_name'] = exploded_df['parsed_items']
    
    # Set price/cost of exploded items to 0 (value is in the combo)
    exploded_df[['price', 'cost']] = 0

    # Combine the processed combo rows with the original non-combo rows
    final_df = pd.concat([non_combo_df, exploded_df], ignore_index=True)
    final_df.drop(columns=['parsed_items'], inplace=True)
    
    logger.info(f"Row count after exploding combos: {len(final_df)}")
    return final_df