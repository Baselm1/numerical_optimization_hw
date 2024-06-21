########################################
######## UTILITY FUNCTIONS #############
########################################


import pandas as pd 

'''
Description: 
    Checks if the values of the left array have a cross over the right array.
    A crossover is defined when we have two lines, for example Fast Moving Average and Slow Moving Average, both plot a line each of various values. However, sometimes we're interested in finding where do they cross each other.
    Having the condition x>y or y>x is not sufficient for generating signals since these conditions hold almost always.
    Instead, for every i'th place in the supplied series, we look at (i-1)'th place and (i+1)'th places to find if x<y and then y<x respectively. 

    NOTE that the same is true for crosses under, but with reversed conditions. However we define the function just for ease of use since the name is verbose and self explanatory.  
'''

def crossed_over(x, y):
    return (x.shift(1) < y.shift(1)) & (x > y)

def crossed_below(x, y):
    return (x.shift(1) > y.shift(1)) & (x < y)


'''

def test_crossover_plot():
    # Load the data
    df = pd.read_parquet('data/ETHUSDT/ETHUSDT-1m-2018-01.parquet').head(200)
    
    # Ensure 'open_time' is in datetime format
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Set 'open_time' as index
    df.set_index('open_time', inplace=True)
    
    # Add fast and slow moving averages
    df['fast_sma'] = ta.sma(df['close'], length=10)
    df['slow_sma'] = ta.sma(df['close'], length=50)
    
    # Find crossover points
    crossover_up = crossed_over(df['fast_sma'], df['slow_sma'])
    crossover_down = crossed_below(df['fast_sma'], df['slow_sma'])

    df['cross_up'] = crossover_up
    df['cross_down'] = crossover_down

    # Plotting
    plt.figure(figsize=(14, 8), dpi=100)
    
    plt.plot(df.index, df['fast_sma'], label='Fast SMA (10)', color='blue')
    plt.plot(df.index, df['slow_sma'], label='Slow SMA (30)', color='red')
    
    # Plot crossover points
    plt.scatter(df.index[crossover_up], df['fast_sma'][crossover_up], marker='^', color='green', s=100, label='Crossover Up')
    plt.scatter(df.index[crossover_down], df['fast_sma'][crossover_down], marker='v', color='orange', s=100, label='Crossover Down')
    plt.title('SMA Crossover Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    return df.drop(columns=['open','high','low','close'])

# Call the test function
t = test_crossover_plot()

pd.set_option('display.max_rows', None)
print(t)

plt.show()

'''