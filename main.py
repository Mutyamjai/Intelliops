from convertion import create_daily_order
from convertion import orders_per_day
from convertion import get_weekly_aggregation
from model import main_ml_pipeline
def main():
    shopify_df = load_and_convert(r"C:\Users\Mutyam Jai\Desktop\python_ai\data\shopify_orders.csv","shopify")
    amazon_df = load_and_convert(r"C:\Users\Mutyam Jai\Desktop\python_ai\data\amazon_orders.csv","amazon")

    final_df = shopify_df._append(amazon_df,ignore_index = True)._append(manual_df,ignore_index = True)
    daily_order_df = create_daily_order(final_df)
    df = get_weekly_aggregation(final_df)
    df = main_ml_pipeline(final_df)
if __name__ == "__main__":
    main()