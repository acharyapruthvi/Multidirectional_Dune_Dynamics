import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math 
import numpy as np

def calculate_wind_direction(um, vm): # Function to calculate wind direction
    ################################################################
    # This function calculates the wind direction for a given list of 
    # meridonal wind and zonal wind directions
    ################################################################
    return (math.degrees(math.atan2(um, vm)) + 360) % 360

def convert_time_to_seconds(t):
    ################################################################
    # This function converts the HH:MM:SS to seconds
    ################################################################
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

def format_func(value, tick_number):
    ################################################################
    # This function converts the seconds to HH:MM:SS format
    ################################################################
    hours, remainder = divmod(int(value), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

# Setting the directory to read the data from
directory = '/Users/pruthviacharya/Desktop/SCP_LANDING/HELLAS_BASIN_CODE/East_Hellespontus'
csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")] # Assuming the csv files are in the directory variable and have the name of ##.csv where LS_##.csv is the ls date

# Creating empty list to store the data 
wind_speed_data = []

# Looping over all of the csv files in the csv_files list
for csv in csv_files:
    ## Reading the csv file
    path = os.path.join(directory, csv) # Setting the path variable to the path of the csv file 
    df = pd.read_csv(path) # Reading the csv file

    # Calcuating the wind direction
    df['Wind Direction'] = df.apply(lambda row: calculate_wind_direction(row['Um'], row['Vm']), axis=1)

    # Removing the LS prefix from the csv files
    date = int(csv[3:-7])

    # Creating a new list using the date and the wind direction
    wind_speed_data.extend(
        [
            {'time_hhmmss': df['time_hhmmss'][ii],
            'Ls': date, 
            'Shear_Stress': df['shear_rate'][ii], 
            'Wind_Direction': df['Wind Direction'][ii]
            } for ii in range(len(df['Wind Direction']))
        ]
    )
# Converting the list into a dataframe 
df_wind_speed = pd.DataFrame(wind_speed_data)
# Sorting the data with increasing Ls date
df_wind_speed = df_wind_speed.sort_values(by=['Ls'], ascending=False)
# Filtering out rows with shear stress less than 0.01 based on the threshold values
df_wind_speed = df_wind_speed[df_wind_speed['Shear_Stress'] >= 0.01]

# Converting the time to seconds
df_wind_speed['time_seconds'] = df_wind_speed['time_hhmmss'].apply(convert_time_to_seconds)

# Calculating the moving average max, min, and mean with a window of 50
window_size = 50 
df_wind_speed['Rolling_Min'] = df_wind_speed['Wind_Direction'].rolling(window=window_size, min_periods=int(window_size/2)).min()
df_wind_speed['Rolling_Max'] = df_wind_speed['Wind_Direction'].rolling(window=window_size, min_periods=int(window_size/2)).max()
df_wind_speed['Rolling_Mean'] = df_wind_speed['Wind_Direction'].rolling(window=int(window_size*1.5), min_periods=int(window_size/2)).mean()

###### Plotting the data #####################
fig, ax = plt.subplots()

# Creating the scatter plot where the size represents the shear stress
sc = ax.scatter(
    df_wind_speed['Ls'],
    df_wind_speed['Wind_Direction'],
    s=df_wind_speed['Shear_Stress'] * 2000,
    c=df_wind_speed['time_seconds'],
    cmap='coolwarm'
)

# Adding the various moving averages to the scatter plot
ax.plot(df_wind_speed['Ls'], df_wind_speed['Rolling_Min'], label='Rolling Min', color='black', linestyle='dotted')
ax.plot(df_wind_speed['Ls'], df_wind_speed['Rolling_Max'], label='Rolling Max', color='tab:green', linestyle='--')
ax.plot(df_wind_speed['Ls'], df_wind_speed['Rolling_Mean'], label='Rolling Mean', color='tab:red')

# Calculate ticks for each hour (assuming your data covers 24 hours)
hour_ticks = np.linspace(0, 23 * 3600, 24)  # 24 ticks for 24 hours

# Creating a custom color bar that displays time in HH:00 format
cbar = plt.colorbar(sc, ax=ax, ticks=hour_ticks)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Custom legend for the scatter plot
sizes = np.linspace(
    np.min(df_wind_speed['Shear_Stress']),
    np.max(df_wind_speed['Shear_Stress']),
    8  # Number of points you want to display on the legend
)
sizes = np.around(sizes, 2)
for size in sizes:
    ax.scatter([], [], s=size * 2000, c="black", label=f'Shear Stress {size}')
ax.legend(loc='best')

# Setting the axes label
plt.xlabel('Ls')
plt.ylabel('Wind Direction [°E]')
plt.title('Wind Direction vs Ls throughout the day with a threshold of 0.01 Pa')
plt.show()
