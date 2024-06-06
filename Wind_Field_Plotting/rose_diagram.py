import re
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import math
from datetime import datetime
import seaborn as sns
from windrose import WindroseAxes
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 10})  

# Function to calculate wind direction
def calculate_wind_direction(um, vm):
    return (math.degrees(math.atan2(um, vm)) + 360) % 360

# Function to calculate wind speed
def calculate_wind_speed(um, vm):
    return np.sqrt(um ** 2 + vm ** 2)

def process_combined_dataframe(df, filenames, show_colorbar, show_cardinal, ls_range, save_directory):
    if 'Um' in df.columns and 'Vm' in df.columns:
        df['Wind Direction'] = df.apply(lambda row: calculate_wind_direction(row['Um'], row['Vm']), axis=1)
        df['Wind_Speed'] = calculate_wind_speed(df['Um'], df['Vm']).round(0)
        
        num_directions = 30 # Divide the 360 direction into 30 equal bins
        df["bin"] = pd.cut(df["Wind Direction"], bins=num_directions, labels=range(num_directions)) # Create a new column with the binned wind direction
        sums = df.groupby("bin")["Wind_Speed"].sum() # Calculate the sum of wind speeds in each bin
        sums_percentage = (sums / sums.sum()) * 100 # Calculate the percentage of wind speeds in each bin
        
        fig = plt.figure(figsize=(1.3*2.125, 1.3*2.7)) # Set the size of the figure
        ax = WindroseAxes.from_ax(fig=fig) # Create a windrose plot
        max_wind_speed = 24 # Limit the maximum wind speed to 24 m/s
        bins = np.linspace(0, max_wind_speed, 10) # Set the bins for the wind speed histogram
        bins = [round(elem, 0) for elem in bins ] # Round the bins to the nearest integer
        bars = ax.bar(df['Wind Direction'], df['Wind_Speed'], normed=True, bins=bins, cmap=cm.gist_rainbow, nsector=num_directions,opening=1) # Plot the windrose plot
        
        if not show_cardinal:
            ax.set_xticklabels([])  # Hide cardinal direction labels

        if show_colorbar:
            cmap = cm.gist_rainbow
            boundaries = bins
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
            ax_cbar = fig.add_axes([0.1, 0.15, 0.8, 0.02]) # Modify these values as needed
            cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=boundaries, ticks=bins, spacing='uniform', orientation='horizontal')
            cbar.set_label(r'Wind Speed [$\frac{m}{s}$]', fontsize=10)  # Setting font size for the colorbar's label
            cbar.ax.xaxis.set_tick_params(labelsize=10)  # Setting font size for the colorbar's tick labels
        # Generate filename based on ls_range
        filename = f'windspeed_{ls_range[0]}_{ls_range[1]}.svg'
        plt.tight_layout()
        full_path = os.path.join(save_directory, filename) 
        plt.savefig(full_path, format='svg')  # Save the figure as SVG
        plt.show()
    else:
        print("Required columns (Um, Vm, shear_rate) missing in the dataframe")

folder_path = ["/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/41.4961, 44.6137", '/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/East_Hellespontus'] # Where the csv files are located
save_folder = ["/home/pruthvi/Desktop/HELLAS_SIMS/CODE/SITE_2", "/home/pruthvi/Desktop/HELLAS_SIMS/CODE/East_Hellespontus"] # Where the figures will be saved

# List of (start_ls, end_ls) tuples
ls_ranges = [(0, 90), (91, 180), (181, 270), (271, 360), (286.4, 348.9), (348.9, 11), (11, 208.2), (208.2, 269.4), (269.4, 311.9), (311.9, 7.7)] # The ranges of Ls values to plot

for ii in range(len(folder_path)): # Loop through the folders
    for start_ls, end_ls in ls_ranges: # Loop through the Ls ranges
        show_colorbar = (start_ls, end_ls) in [(0, 90), (286.4, 348.9)] # Show the colorbar only for the specified Ls ranges
        show_cardinal = not show_colorbar # Hide the cardinal direction labels only for the specified Ls ranges
        dfs = [] # List of dataframes
        valid_files = [] # List of valid filenames
        for i in range(0, 360, 10): # Loop through the Ls values
            ls_value = i
            if start_ls <= end_ls:
                in_range = start_ls <= ls_value <= end_ls
            else:
                in_range = ls_value >= start_ls or ls_value <= end_ls
            if in_range:
                file_name = f"LS_{i}.nc.csv" # filename based on the Ls value
                full_path = os.path.join(folder_path[ii], file_name)
                if os.path.exists(full_path):
                    df = pd.read_csv(full_path) # Read the csv file
                    dfs.append(df) # Append the dataframe to the list of dataframes
                    valid_files.append(file_name) # Append the filename to the list of valid filenames

        if dfs:
            concatenated_df = pd.concat(dfs, ignore_index=True) # Concatenate the dataframes
            process_combined_dataframe(concatenated_df, valid_files, show_colorbar, show_cardinal, (start_ls, end_ls),save_directory=save_folder[ii]) # Plot the windrose plot for the specified Ls range
        else:
            print(f"No data files found in the specified Ls range: {start_ls}-{end_ls}.") # Print an error message if no data files are found in the specified Ls range
