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

    # # Resample to 15-minute intervals (adjust if needed)
    # df_resampled = df.resample('1min', on='open_time').agg({
    #     'open': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'close': 'last'
    # })

    # # Ensure all columns are properly aggregated
    # df = df_resampled.reset_index()

    # Ensure the column names are in lower case to match your data labels
    df.columns = ['open_time', 'open', 'high', 'low', 'close']

    # Step 2: Calculate Ichimoku Cloud components
    # Calculate Ichimoku Cloud components with specified parameters
    ichimoku_cloud, _ = ta.ichimoku(high=df['high'], low=df['low'], close=df['close'],
                                    tenkan=tenkan_sen_length, kijun=kijun_sen_length, senkou=senkou_span_length,
                                    include_chikou=False)

    # Extract individual components
    tenkan_sen = ichimoku_cloud[f'ITS_{tenkan_sen_length}']
    kijun_sen = ichimoku_cloud[f'IKS_{kijun_sen_length}']
    span_a = ichimoku_cloud[f'ISA_{tenkan_sen_length}']  # Adjust for span A
    span_b = ichimoku_cloud[f'ISB_{kijun_sen_length}']

    # Step 3: Define Strategy Logic
    # Create signals DataFrame
    signals = pd.DataFrame(index=df.index)

    # Define Strategy Logic based on extracted components
    # Long when tenkan_sen crosses over kijun_sen, close is above both span A and span B
    long_entries = crossed_over(tenkan_sen, kijun_sen) & (df['close'] > span_a) & (df['close'] > span_b)

    # Short when tenkan_sen crosses below kijun_sen, close is below both span A and span B
    short_entries = crossed_below(tenkan_sen, kijun_sen) & (df['close'] < span_a) & (df['close'] < span_b)

    # Exit conditions for long positions: tenkan_sen < kijun_sen
    long_exits = crossed_below(tenkan_sen, kijun_sen)

    # Exit conditions for short positions: tenkan_sen > kijun_sen
    short_exits = crossed_over(tenkan_sen, kijun_sen)

    # Assign strategy signals to the signals DataFrame
    signals['long'] = long_entries.astype(int)
    signals['short'] = short_entries.astype(int)
    signals['long_exit'] = long_exits.astype(int)
    signals['short_exit'] = short_exits.astype(int)

    # Step 4: Backtest the Strategy with vectorbt
    # Create portfolio
    print(f'Params: {tenkan_sen_length}, {kijun_sen_length}, {senkou_span_length}')
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        high=df['high'],
        low=df['low'],
        entries=signals['long'],
        exits=signals['long_exit'],
        short_entries=signals['short'],
        short_exits=signals['short_exit'],
        freq='1m',  # Frequency for closing positions (adjust as needed)
        fees=0.001
    )

    # Calculate portfolio statistics
    stats = portfolio.stats()

    return stats

def optimize_strategy_worker(params):
    data, tenkan_sen_length, kijun_sen_length, senkou_span_length = params
    print()
    return (tenkan_sen_length, kijun_sen_length, senkou_span_length), backtest_strategy(data, tenkan_sen_length, kijun_sen_length, senkou_span_length)

def optimize_strategy(data):
    parameter_grid = {
        'tenkan_sen_length': range(5, 51, 2),    # Adjust range and step size as needed
        'kijun_sen_length': range(15, 46, 2),
        'senkou_span_length': range(25, 56, 2)
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
    print("Reading data...")
    paths = [os.path.join('data/DOGEUSDT', file) for file in os.listdir('data/DOGEUSDT') if file.endswith('.parquet')]
    data = concat_parquet_files(paths)

    print('Optimizing strategy...')
    start = time.time()
    results = optimize_strategy((data.head(100000)).tail(int(len(data)/15)))
    print('Done optimization')

    best_params, best_stats = max(results.items(), key=lambda x: x[1]['Total Return [%]'])

    print('\n')
    print("---------------------")
    print("Best Parameters:", best_params)
    print("Best Stats:")
    print(best_stats)

    print(f'Total time taken: {time.time() - start} seconds')
    print('\n')
    print("---------------------")
    plot_surface_with_contours(results, metric='Total Return')

