import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from datetime import datetime

# Import the functions you want to test
from reporting.data_preparation import clean_data_for_reporting, explode_combo_items,explode_combo_items_advanced
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

@pytest.fixture
def sample_combo_data():
    """Provides sample data with a complex combo for testing."""
    data = {
        'receipt_number': ['1-1696', '1-1696', '1-1695'],
        'item_name': ['Combo Pa ́ Dos', 'Malteada Chocolate', 'Doble Chicken'],
        'price': [240.0, 60.0, 115.0],
        'cost': [129.0, 19.0, 54.0],
        'modifiers': [
            'Hamburguesa 1(Hamburguesa Smash 1);Hamburguesa 2(Hamburguesa Chiken 2);Mayonesa(Ajo);Mayonesa(Chipotle);Refresco Sabor(Agua Natural)',
            'Tipo de Leche(Leche Entera)',
            'Mayonesa(Ajo)'
        ]
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

def test_explode_combo_items_advanced(sample_combo_data):
    """
    Tests the advanced combo exploding function to ensure it correctly unnests
    items, renames them, and associates the correct modifiers.
    """
    # Run the function on the sample data
    result_df = explode_combo_items_advanced(sample_combo_data)

    # --- ASSERTIONS ---

    # 1. Check that the total number of rows is correct
    # (2 original non-combo rows + 3 new rows from the exploded combo)
    assert len(result_df) == 5, "The final DataFrame should have 5 rows."

    # 2. Check that the original combo row is gone
    assert 'Combo Pa ́ Dos' not in result_df['item_name'].values

    # 3. Verify the new rows were created correctly
    smash_burger_row = result_df[result_df['item_name'] == 'Smash Burger']
    chicken_burger_row = result_df[result_df['item_name'] == 'Chicken Burger']
    agua_row = result_df[result_df['item_name'] == 'Agua Natural']

    # Check that each new item exists
    assert not smash_burger_row.empty, "Smash Burger row was not created."
    assert not chicken_burger_row.empty, "Chicken Burger row was not created."
    assert not agua_row.empty, "Agua Natural row was not created."

    # 4. Verify that modifiers were correctly associated
    assert smash_burger_row.iloc[0]['modifiers'] == 'Mayonesa(Ajo)', "Smash Burger should have 'Mayonesa(Ajo)'."
    assert chicken_burger_row.iloc[0]['modifiers'] == 'Mayonesa(Chipotle)', "Chicken Burger should have 'Mayonesa(Chipotle)'."
    assert agua_row.iloc[0]['modifiers'] is None, "Agua Natural should have no modifier."

    # 5. Verify that the price and cost of new items are zero
    assert smash_burger_row.iloc[0]['price'] == 0
    assert chicken_burger_row.iloc[0]['price'] == 0
    assert agua_row.iloc[0]['price'] == 0
