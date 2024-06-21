import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from src.utils import crossed_below, crossed_over
import os
import time
import multiprocessing
import plotly.graph_objects as go
import numpy as np

def backtest_strategy(data, tenkan_sen_length, kijun_sen_length, senkou_span_length):
    # Step 1: Read the Parquet file
    df = data.copy()

    # Convert 'open_time' to datetime assuming it's in milliseconds
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    # Resample to 1-minute intervals (adjust if needed)
    df_resampled = df.resample('15min', on='open_time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    # Ensure all columns are properly aggregated
    df = df_resampled.reset_index()

    # Ensure the column names are in lower case to match your data labels
    df.columns = ['open_time', 'open', 'high', 'low', 'close']

    # Step 2: Calculate Ichimoku Cloud components
    ichimoku_cloud, _ = ta.ichimoku(high=df['high'], low=df['low'], close=df['close'],
                                    tenkan=tenkan_sen_length, kijun=kijun_sen_length, senkou=senkou_span_length,
                                    include_chikou=False)

    tenkan_sen = ichimoku_cloud[f'ITS_{tenkan_sen_length}']
    kijun_sen = ichimoku_cloud[f'IKS_{kijun_sen_length}']
    span_a = ichimoku_cloud[f'ISA_{tenkan_sen_length}']
    span_b = ichimoku_cloud[f'ISB_{kijun_sen_length}']

    # Step 3: Calculate EMA and SMA
    ema = ta.ema(df['close'], length=50)
    sma = ta.sma(df['close'], length=100)

    # Calculate ATR for stop loss, take profit, and trailing stop
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)

    long_entries = (crossed_over(tenkan_sen, kijun_sen) & 
                    (df['close'] > span_a) & 
                    (df['close'] > span_b) &
                    (df['close'] > ema) & 
                    (df['close'] > sma))

    # Step 5: Backtest the Strategy with vectorbt
    stop_loss = atr * 2  # Example: Stop loss at 2 times the ATR value
    take_profit = atr * 4  # Example: Take profit at 4 times the ATR value
    trailing_stop = atr * 1.5  # Example: Trailing stop at 1.5 times the ATR value

    long_exits = crossed_below(tenkan_sen, kijun_sen)

    print(f'Params: {tenkan_sen_length}, {kijun_sen_length}, {senkou_span_length}')
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        high=df['high'],
        low=df['low'],
        entries=long_entries,
        exits=long_exits,
        freq='15min',
        fees=0.001,
        sl_stop=stop_loss,  # Stop loss
        tp_stop=take_profit,  # Take profit
        sl_trail=trailing_stop,  # Trailing stop
        use_stops=True,
        init_cash=10000,
        size=1
    )

    stats = portfolio.stats()
    portfolio.plot().show()

    return stats

def crossed_over(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 > series2)

def crossed_below(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 < series2)



def optimize_strategy_worker(params):
    data, tenkan_sen_length, kijun_sen_length, senkou_span_length = params
    return (tenkan_sen_length, kijun_sen_length, senkou_span_length), backtest_strategy(data, tenkan_sen_length, kijun_sen_length, senkou_span_length)

def optimize_strategy(data):
    parameter_grid = {
        'tenkan_sen_length': range(5, 51, 3),    # Adjust range and step size as needed
        'kijun_sen_length': range(20, 61, 3),
        'senkou_span_length': range(30, 91, 3)
    }

    results = {}
    params_list = [(data, t, k, s) for t in parameter_grid['tenkan_sen_length']
                   for k in parameter_grid['kijun_sen_length']
                   for s in parameter_grid['senkou_span_length']]

    # Use multiprocessing Pool to parallelize
    with multiprocessing.Pool() as pool:
        for params, result in pool.imap_unordered(optimize_strategy_worker, params_list):
            results[params] = result

    return results

def plot_scatter(results, metric='Sharpe Ratio'):
    # Extract parameter values and metric values from results
    tenkan_sen_values = [key[0] for key in results.keys()]
    kijun_sen_values = [key[1] for key in results.keys()]
    senkou_span_values = [key[2] for key in results.keys()]
    
    # Extract metric values
    metric_values = []
    for key in results.keys():
        stats = results[key]
        if metric == 'Sharpe Ratio':
            metric_value = stats['Sharpe Ratio']
        elif metric == 'Total Return':
            metric_value = stats['Total Return']
        else:
            metric_value = None  # Handle additional metrics if needed
        metric_values.append(metric_value)
    
    # Create scatter plot using Plotly
    fig = go.Figure(data=go.Scatter3d(
        x=tenkan_sen_values,
        y=kijun_sen_values,
        z=senkou_span_values,
        mode='markers',
        marker=dict(
            size=12,
            color=metric_values,
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8,
            colorbar=dict(title=metric)
        )
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Tenkan-sen Length',
            yaxis_title='Kijun-sen Length',
            zaxis_title='Senkou Span Length'
        ),
        title=f'Optimization Scatter Plot ({metric})'
    )
    
    fig.show()

def plot_surface_with_contours(results, metric='Sharpe Ratio'):
    # Extract parameter values and metric values from results
    parameter_combinations = list(results.keys())
    tenkan_sen_values = sorted(set([params[0] for params in parameter_combinations]))
    kijun_sen_values = sorted(set([params[1] for params in parameter_combinations]))
    
    # Extract metric values
    metric_values = []
    for params in parameter_combinations:
        stats = results[params]
        if metric == 'Sharpe Ratio':
            metric_value = stats['Sharpe Ratio']
        elif metric == 'Total Return':
            metric_value = stats.get('Total Return [%]')  # Use get() to handle missing keys gracefully
        else:
            metric_value = None  # Handle additional metrics if needed
        metric_values.append(metric_value)
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(tenkan_sen_values, kijun_sen_values)
    
    # Ensure Z is initialized with float type for NaNs
    Z = np.zeros_like(X, dtype=np.float64)
    
    # Populate Z with metric values
    for i, params in enumerate(parameter_combinations):
        tenkan_index = tenkan_sen_values.index(params[0])
        kijun_index = kijun_sen_values.index(params[1])
        if metric_values[i] is not None:
            Z[kijun_index, tenkan_index] = metric_values[i]
    
    # Create Surface plot using Plotly
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True,
            highlightwidth=4,
        ),
        colorscale='Viridis'  # Choose a colorscale
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Tenkan-sen Length',
            yaxis_title='Kijun-sen Length',
            zaxis_title='Metric Value' if metric else 'Senkou Span Length',
            aspectmode='manual',  # Ensure manual aspect ratio control
            aspectratio=dict(x=2, y=1, z=0.5),  # Adjust as needed
        ),
        title=f'Optimization Surface Plot ({metric})',
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=100),  # Adjust margins as needed
        template='plotly_dark'  # Set dark mode theme
    )
    
    fig.show()

def concat_parquet_files(file_paths):
    dfs = []

    # Iterate through each file path
    for file_path in file_paths:
        # Read the parquet file
        df = pd.read_parquet(file_path)

        # Convert 'open_time' to datetime assuming it's in milliseconds
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list along rows
    concatenated_df = (pd.concat(dfs, ignore_index=True)).sort_values(by='open_time', ascending=True)
    concatenated_df.reset_index(drop=True, inplace=True)
    return concatenated_df

if __name__ == "__main__":
    # Testing the best parameters but for the entire dataset:
    path = 'data/BTCUSDT'
    print("Reading data...")
    paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.parquet')]
    data = concat_parquet_files(paths)
    print("Backtesting...")
    stats = backtest_strategy(data, tenkan_sen_length=9, kijun_sen_length=26, senkou_span_length=52)
    print(stats)

# .head(100000).tail(50000)