import pandas as pd

#Top 5 products by sales this month
def top_five_products(df):
    """
    Returns the top 5 products by sales for the current month.
    """
    #Confirms month's data is present
    if df.name != f'curated_data_{pd.to_datetime("now").strftime("%Y-%m")}.csv':
        raise ValueError("DataFrame must have the current month's data 'curated_data_<current_month>.csv'")


    monthly_sales = df["item"]
    
    if monthly_sales.empty:
        return pd.DataFrame(columns=['product_id', 'total_sales'])
    
    top_products = monthly_sales.groupby('product_id')['total_price'].sum().nlargest(5).reset_index()
    top_products.columns = ['product_id', 'total_sales']
    
    return top_products