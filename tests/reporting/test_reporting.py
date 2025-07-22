import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from datetime import datetime

# Import the functions you want to test
from reporting.data_preparation import clean_data_for_reporting, explode_combo_items
from reporting.monthly_report import top_five_products

# --- 1. Create Sample Data for Testing ---
@pytest.fixture
def sample_raw_data():
    """Provides a sample DataFrame for tests."""
    data = {
        'receipt_number': ['R1', 'R1', 'R2'],
        'datetime': ['2025-07-21 10:00:00', '2025-07-21 10:00:00', '2025-07-21 11:00:00'],
        'item_name': ['Combo Hamburguesa', 'Refresco', 'Papas Fritas'],
        'price': ['150.0', '25.0', 50.0], # Mix of string and float
        'cost': [70.0, 10.0, 20.0],
        'modifiers': ['Hamburguesa(Carne);Refresco(Coca-Cola)', None, None]
    }
    return pd.DataFrame(data)

# --- 2. Write a Test for Each Function ---

def test_clean_data_for_reporting(sample_raw_data):
    """Tests data type correction, null handling, and feature engineering."""
    cleaned_df = clean_data_for_reporting(sample_raw_data)
    
    # Assert that dtypes are correct
    assert pd.api.types.is_numeric_dtype(cleaned_df['price'])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['datetime'])
    
    # Assert that new feature columns were created
    assert 'day_of_week' in cleaned_df.columns
    assert 'hour_of_day' in cleaned_df.columns

def test_explode_combo_items(sample_raw_data):
    """Tests that combo items are correctly exploded into new rows."""
    exploded_df = explode_combo_items(sample_raw_data)
    
    # Assert that a new row was created for the combo item
    assert len(exploded_df) == 4
    
    # Assert that the new item 'Carne' is now in the item_name column
    assert 'Carne' in exploded_df['item_name'].values
    
    # Assert that the price for the exploded item is 0
    assert exploded_df[exploded_df['item_name'] == 'Carne']['price'].iloc[0] == 0

def test_top_five_products(sample_raw_data):
    """Tests the top 5 products calculation on CLEANED data."""
    
    # --- FIX: Clean the data first, just like in your main script ---
    cleaned_df = clean_data_for_reporting(sample_raw_data)
    
    # Now, call the function with the cleaned DataFrame
    result_df = top_five_products(cleaned_df)
    
    # Define what the correct output should be
    expected_data = {
        'item_name': ['Combo Hamburguesa', 'Papas Fritas', 'Refresco'],
        'total_sales': [150.0, 50.0, 25.0]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # pandas has a built-in function to compare DataFrames in tests
    assert_frame_equal(result_df, expected_df, check_dtype=False)