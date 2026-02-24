import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_time_series_data(final_df):
    df_copy = final_df.copy()
    df_copy['event_time'] = pd.to_datetime(df_copy['event_time'], dayfirst=True, errors='coerce')
    
    # Remove invalid dates
    df_copy = df_copy.dropna(subset=['event_time'])
    
    # Get daily aggregation
    daily_df = df_copy.groupby(df_copy['event_time'].dt.date).agg({
        'order_id': 'count',
        'revenue': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    daily_df.columns = ['date', 'daily_orders', 'daily_revenue', 'daily_quantity']
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Create complete date range
    date_range = pd.date_range(
        start=daily_df['date'].min(),
        end=daily_df['date'].max(),
        freq='D'
    )
    
    # Reindex with forward fill for missing dates (better than 0)
    daily_df = daily_df.set_index('date').reindex(date_range).reset_index()
    daily_df.columns = ['date', 'daily_orders', 'daily_revenue', 'daily_quantity']
    
    # Forward fill small gaps (max 2 days), otherwise use 0
    daily_df['daily_orders'] = daily_df['daily_orders'].fillna(method='ffill', limit=2).fillna(0)
    daily_df['daily_revenue'] = daily_df['daily_revenue'].fillna(method='ffill', limit=2).fillna(0)
    daily_df['daily_quantity'] = daily_df['daily_quantity'].fillna(method='ffill', limit=2).fillna(0)
    
    # Sort by date
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Total days: {len(daily_df)}")
    print(f"Days with sales: {(daily_df['daily_revenue'] > 0).sum()}")
    print(f"Average daily revenue: ${daily_df['daily_revenue'].mean():.2f}")
    
    return daily_df

def create_features(daily_df):
    """
    Create optimized time series features
    """
    df = daily_df.copy()
    
    # Basic time features
    df['day_index'] = range(len(df))
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    
    # Lag features (previous values)
    for lag in [1, 2, 3, 7]:
        df[f'lag_revenue_{lag}'] = df['daily_revenue'].shift(lag)
        df[f'lag_orders_{lag}'] = df['daily_orders'].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [3, 7, 14]:
        df[f'rolling_mean_revenue_{window}'] = df['daily_revenue'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_revenue_{window}'] = df['daily_revenue'].rolling(window=window, min_periods=1).std()
        df[f'rolling_mean_orders_{window}'] = df['daily_orders'].rolling(window=window, min_periods=1).mean()
    
    # Exponential moving average (gives more weight to recent data)
    df['ema_revenue_7'] = df['daily_revenue'].ewm(span=7, adjust=False).mean()
    
    # Change features
    df['revenue_change_1d'] = df['daily_revenue'] - df['lag_revenue_1']
    df['revenue_pct_change_1d'] = df['daily_revenue'].pct_change(1) * 100
    
    # Trend features
    df['revenue_trend_7d'] = df['daily_revenue'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
    )
    
    # Fill NaN values strategically
    # For lag features, use forward fill then backward fill
    lag_columns = [col for col in df.columns if 'lag_' in col]
    for col in lag_columns:
        df[col] = df[col].fillna(method='bfill').fillna(0)
    
    # For rolling features, forward fill
    rolling_columns = [col for col in df.columns if 'rolling_' in col or 'ema_' in col]
    for col in rolling_columns:
        df[col] = df[col].fillna(method='ffill').fillna(0)
    
    # For change features, fill with 0
    df['revenue_change_1d'] = df['revenue_change_1d'].fillna(0)
    df['revenue_pct_change_1d'] = df['revenue_pct_change_1d'].fillna(0)
    df['revenue_trend_7d'] = df['revenue_trend_7d'].fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"\nFeatures created: {len(df.columns)}")
    print(f"Rows with complete data: {df.notna().all(axis=1).sum()}")
    
    return df

def train_linear_regression_model(features_df, target='daily_revenue'):
    df = features_df.copy()
    
    # Select best features (remove highly correlated or redundant ones)
    feature_columns = [
        'day_index',
        'day_of_week',
        'is_weekend',
        'is_month_start',
        'is_month_end',
        
        # Lag features
        'lag_revenue_1',
        'lag_revenue_2',
        'lag_revenue_3',
        'lag_revenue_7',
        'lag_orders_1',
        'lag_orders_3',
        
        # Rolling features
        'rolling_mean_revenue_7',
        'rolling_std_revenue_7',
        'rolling_mean_orders_7',
        'ema_revenue_7',
        
        # Change features
        'revenue_change_1d',
        'revenue_trend_7d'
    ]
    
    # Remove rows where critical lag features are missing
    # Keep rows after day 7 to have proper lag features
    df_clean = df[df['day_index'] >= 7].copy()
    
    # Check if we have enough data
    if len(df_clean) < 20:
        print(f"⚠️ WARNING: Only {len(df_clean)} days of data available. Need at least 20 days for reliable model.")
        print("Consider:")
        print("1. Adding more data")
        print("2. Using simpler model (Moving Average)")
        return None, None, None, None, None, None, None
    
    # Prepare X and y
    X = df_clean[feature_columns].copy()
    y = df_clean[target].copy()
    
    # Check for any remaining NaN or inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"Target mean: ${y.mean():.2f}")
    print(f"Target std: ${y.std():.2f}")
    
    # Split data (80% train, 20% test)
    # Use at least last 20% for test, minimum 5 days
    test_size = max(0.2, 5 / len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features (important for Linear Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train': {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R2': r2_score(y_train, y_train_pred),
            'MAPE': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100 if y_train.mean() > 0 else 0
        },
        'test': {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R2': r2_score(y_test, y_test_pred),
            'MAPE': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100 if y_test.mean() > 0 else 0
        }
    }
    
    # Create results dataframe
    test_indices = df_clean.index[-len(y_test):]
    results_df = df_clean.loc[test_indices].copy()
    results_df['predicted_revenue'] = y_test_pred
    results_df['actual_revenue'] = y_test.values
    results_df['prediction_error'] = results_df['actual_revenue'] - results_df['predicted_revenue']
    results_df['error_pct'] = (results_df['prediction_error'] / results_df['actual_revenue'] * 100).round(2)
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n=== Top 5 Most Important Features ===")
    print(feature_importance.head())
    
    return model, X_train, X_test, y_train, y_test, metrics, results_df, scaler


def main_ml_pipeline(final_df):
    
    print("=" * 60)
    print("SALES FORECASTING ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Prepare data
    print("\n[1/4] Preparing time series data...")
    daily_df = prepare_time_series_data(final_df)
    
    if len(daily_df) < 20:
        print("\n❌ ERROR: Not enough data for ML model!")
        print("Need at least 20 days of data. Consider using Moving Average instead.")
        return None
    
    # Step 2: Create features
    print("\n[2/4] Creating features...")
    features_df = create_features(daily_df)
    
    # Step 3: Train model
    print("\n[3/4] Training Linear Regression model...")
    result = train_linear_regression_model(features_df, target='daily_revenue')
    
    if result[0] is None:
        print("\n❌ Model training failed!")
        return None
    
    model, X_train, X_test, y_train, y_test, metrics, results_df, scaler = result
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"Training Set:")
    print(f"  MAE:  ${metrics['train']['MAE']:.2f}")
    print(f"  RMSE: ${metrics['train']['RMSE']:.2f}")
    print(f"  R²:   {metrics['train']['R2']:.4f}")
    print(f"  MAPE: {metrics['train']['MAPE']:.2f}%")
    
    print(f"\nTest Set:")
    print(f"  MAE:  ${metrics['test']['MAE']:.2f}")
    print(f"  RMSE: ${metrics['test']['RMSE']:.2f}")
    print(f"  R²:   {metrics['test']['R2']:.4f}")
    print(f"  MAPE: {metrics['test']['MAPE']:.2f}%")
    
    # Interpret results
    if metrics['test']['R2'] < 0:
        print("\n WARNING: Negative R² indicates poor model fit!")
    elif metrics['test']['R2'] < 0.5:
        print("\n Moderate model performance. Consider Moving Average as alternative.")
    else:
        print("\n Good model performance!")
    
    # Step 4: Moving Average (always show as comparison)
    print("\n[4/4] Moving Average forecast (for comparison)...")
    ma_forecast_df = moving_average_forecast(features_df, window=7, target='daily_revenue')
    
    print("\n=== Sample Predictions ===")
    print(results_df[['date', 'actual_revenue', 'predicted_revenue', 'error_pct']].head(10))
    
    print("\n" + "=" * 60)
    
    return {
        'daily_df': daily_df,
        'features_df': features_df,
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'results_df': results_df,
        'ma_forecast_df': ma_forecast_df
    }


def moving_average_forecast(features_df, window=7, target='daily_revenue'):
   
    df = features_df.copy()
    
    df['ma_forecast'] = df[target].rolling(window=window, min_periods=1).mean().shift(1)
    df = df[df['day_index'] >= window].reset_index(drop=True)
    
    df['forecast_error'] = df[target] - df['ma_forecast']
    df['forecast_error_pct'] = (df['forecast_error'] / df[target] * 100).round(2)
    
    mae = mean_absolute_error(df[target], df['ma_forecast'])
    rmse = np.sqrt(mean_squared_error(df[target], df['ma_forecast']))
    mape = np.mean(np.abs(df['forecast_error_pct']))
    
    print(f"\n=== Moving Average Forecast (window={window}) ===")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return df[['date', target, 'ma_forecast', 'forecast_error', 'forecast_error_pct']]