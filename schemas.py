import pyarrow as pa


# Define the schema to enforce when saving/loading parquet files
DEFINED_SCHEMA = pa.schema([
    pa.field('receipt_number', pa.string()),
    pa.field('datetime', pa.timestamp('ns')),
    pa.field('date', pa.string()),
    pa.field('time', pa.string()),
    pa.field('order_type', pa.string()),
    pa.field('item_name', pa.string()),
    pa.field('cost', pa.float64()),
    pa.field('price', pa.float64()),
    pa.field('total_money', pa.float64()),
    pa.field('modifiers', pa.string()),
    pa.field('payment_type', pa.string()),
    pa.field('shifted_time', pa.timestamp('ns')),
    pa.field('minutes_past_midnight', pa.int64()),
    pa.field('time_slot', pa.string()), # Store categorical as string for stability
])