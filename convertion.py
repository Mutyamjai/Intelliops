import pandas as pd

def create_daily_order(final_df):
    df = final_df.sort_values(by='event_time')
    mapped = {}
    for i, row in df.iterrows():
        event_time = row.get("event_time")
        revenue = row.get("revenue")
        
        if event_time in mapped:
            mapped[event_time] += revenue
        else:
            mapped[event_time] = revenue

    df = pd.DataFrame(mapped.items(),columns=['date','revenue_per_day'])
    df["change"] = df["revenue_per_day"] - df["revenue_per_day"].shift(1)
    df["change_%"] = (df["change"]/df["revenue_per_day"])*100
    df["change_%"] = df["change_%"].round(decimals = 2)
    
def orders_per_day(final_df):
    df = final_df.groupby("event_time").agg(
        no_of_orders = ('order_id','count')
    ).sort_values(by="no_of_orders", ascending=False).reset_index()
    print(df)
    return df

def revenue_per_product(final_df):
    df = final_df.groupby("product").agg(
        product_revenue = ('revenue','sum')
    ).sort_values(by="product_revenue", ascending=False).reset_index()
    print(df)
    return df


def quantity_per_product(final_df):
    quantity_per_product = final_df.groupby("product")["quantity"].sum().sort_values(by="quantity", ascending=False).reset_index()

    print(quantity_per_product)
    return quantity_per_product

def summary_of_product(final_df):
    product_summary = final_df.groupby("product").agg(
        total_revenue=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        total_orders=("order_id", "nunique")
    ).sort_values(by="total_revenue", ascending=False).reset_index()
    print(product_summary)
    return product_summary

def get_daily_aggregation(final_df):
    daily_summary = final_df.groupby("event_time").agg(
        total_revenue=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        total_orders=("order_id", "nunique")
    ).sort_values(by="event_time", ascending=False).reset_index()
    print(daily_summary)
    return daily_summary

def get_weekly_aggregation(df):
    # Convert event_time to datetime with dayfirst=True
    df['event_time'] = pd.to_datetime(df['event_time'], dayfirst=True, errors='coerce')
    
    # Extract week start date (Monday of each week)
    df['week_start'] = df['event_time'].dt.to_period('W').dt.start_time
    
    # Group by week and aggregate
    weekly_agg = df.groupby('week_start').agg({
        'order_id': 'count',
        'revenue': 'sum',
        'quantity': 'sum'
    }).reset_index()
    print(weekly_agg)
    # Add week end date
    weekly_agg['week_end'] = weekly_agg['week_start'] + pd.Timedelta(days=6)
    
    # Rename and reorder columns
    weekly_agg.columns = ['week_start', 'total_orders', 'total_revenue', 'total_quantity', 'week_end']
    weekly_agg = weekly_agg[['week_start', 'week_end', 'total_orders', 'total_revenue', 'total_quantity']]
    print(weekly_agg)
    return weekly_agg.sort_values('week_start').reset_index(drop=True)

def get_rolling_7day_average(df):
    # Get daily aggregation first
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['event_time']).dt.date
    
    # Daily aggregation
    daily_agg = df_copy.groupby('date').agg({
        'order_id': 'count',
        'revenue': 'sum'
    }).reset_index()
    
    daily_agg.columns = ['date', 'daily_orders', 'daily_revenue']
    
    # Sort by date to ensure correct rolling calculation
    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
    
    # Calculate 7-day rolling average
    daily_agg['rolling_avg_orders'] = daily_agg['daily_orders'].rolling(window=7, min_periods=1).mean()
    daily_agg['rolling_avg_revenue'] = daily_agg['daily_revenue'].rolling(window=7, min_periods=1).mean()
    
    # Round for readability
    daily_agg['rolling_avg_orders'] = daily_agg['rolling_avg_orders'].round(2)
    daily_agg['rolling_avg_revenue'] = daily_agg['rolling_avg_revenue'].round(2)
    
    return daily_agg[['date', 'rolling_avg_orders', 'rolling_avg_revenue']]