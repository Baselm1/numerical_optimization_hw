import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from src.utils import crossed_below, crossed_over
import os
import time
import multiprocessing
import plotly.graph_objects as go
import numpy as np

import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import os
import multiprocessing
import plotly.graph_objects as go
import numpy as np

def backtest_strategy(data, tenkan_sen_length, kijun_sen_length, senkou_span_length):
    df = data.copy()

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    df_resampled = df.resample('15min', on='open_time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    df = df_resampled.reset_index()
    df.columns = ['open_time', 'open', 'high', 'low', 'close']

    # Generate a long and short signal at every candle
    entries = pd.Series(True, index=df.index)
    
    # Generate exit signals 10 candles after each entry
    exits = entries.shift(10).fillna(False)
    
    # Calculate stop loss and take profit prices
    long_stop_loss = df['close'] * 0.98
    long_take_profit = df['close'] * 1.05
    short_stop_loss = df['close'] * 1.02
    short_take_profit = df['close'] * 0.95
    
    # Calculate order size as 5% of the initial cash
    order_size = 0.05

    # Create the portfolio
    portfolio = vbt.Portfolio.from_orders(
        close=df['close'],
        size=np.where(entries, order_size, -order_size),  # Long and short position sizes
        price=df['close'],  # Entry price
        fees=0.001,
        init_cash=10000,
        freq='15min'
        )

    stats = portfolio.stats()
    portfolio.plot().show()

    return stats

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
    stats = backtest_strategy(data.tail(10000), tenkan_sen_length=9, kijun_sen_length=26, senkou_span_length=52)
    print(stats)
    

# .head(100000).tail(50000)