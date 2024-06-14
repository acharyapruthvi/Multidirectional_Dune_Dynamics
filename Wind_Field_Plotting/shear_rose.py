import re
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Required matplotlib backend for my linux machine
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import math
from datetime import datetime
import seaborn as sns
from windrose import WindroseAxes
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
import csv
##############################################################################################################
# Script plots the shear rate and the wind direction from the MMM model outputs
# Colours show the range of the shear rate for the Ls range used for Figure 4 and the supplimentary figures
##############################################################################################################

plt.rcParams.update({'font.size': 10})  


def calculate_wind_direction(um, vm): # Function to calculate wind direction
    return (math.degrees(math.atan2(um, vm)) + 360) % 360


def calculate_shear_rate(shear_column): # Calculate shear rate instead of wind speed
    return np.abs(shear_column)

def find_max_min_shear_rate(folder_path):
    """
    Reads all CSV files in the specified folder, extracts the 'shear_rate' column,
    and returns the maximum and minimum shear rates.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        tuple: A tuple containing the maximum and minimum shear rates.
    """
    try:
        max_shear_rate = float('-inf')
        min_shear_rate = float('inf')

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)

                if 'shear_rate' in df.columns:
                    max_shear_rate = max(max_shear_rate, df['shear_rate'].max())
                    min_shear_rate = min(min_shear_rate, df['shear_rate'].min())

        if max_shear_rate != float('-inf') and min_shear_rate != float('inf'):
            return max_shear_rate, min_shear_rate
        else:
            return None, None  # No valid data found in the CSV files

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return None, None

def process_combined_dataframe(df, filenames, show_colorbar, show_cardinal, ls_range, save_directory): # Function to plot the shear rate and the wind direction from the MMM model outputs
    if 'Um' in df.columns and 'Vm' in df.columns and 'shear_rate' in df.columns:
        df['Wind Direction'] = df.apply(lambda row: calculate_wind_direction(row['Um'], row['Vm']), axis=1) # Calculate the wind direction
        df['Shear_Rate'] = calculate_shear_rate(df['shear_rate']) # Extarct the shear rate column

        # Filter out non-finite shear rate values
        df = df[np.isfinite(df['Shear_Rate'])] # Filter out non-finite shear rate values

        if df.empty: # Error message if no valid data found
            print(f"No valid data after filtering for Ls range: {ls_range}")
            return

        num_directions = 30 # Divide 360 direction into 30 equal bins
        df["bin"] = pd.cut(df["Wind Direction"], bins=num_directions, labels=range(num_directions)) # Extract the wind direction column into 30 equal bins
        sums = df.groupby("bin")["Shear_Rate"].sum() # Sum the shear rate values for each bin
        sums_percentage = (sums / sums.sum()) * 100 # Calculata the percentage of the wind in a given direction

        fig = plt.figure(figsize=(1.2 * 2.125, 1.2 * 2.7)) # Figure size
        ax = WindroseAxes.from_ax(fig=fig) # Set up the windrose plot

        bins = np.linspace(min_shear, max_shear, 6) # Divide the shear rate into 6 equal bins
        bins = [round(elem, 8) for elem in bins]  # Adjusted precision to match your data range

        if len(set(bins)) == 1:  # All bins are the same
            bins = [bins[0] - 0.1, bins[0] + 0.1]

        # Custom RGB colors
        custom_colors = [
            (1, 112, 254),        # blue (lowest bound)
            (97, 254, 30),        # green   |
            (255, 255, 1),        # yellow  | 
            (255, 214, 33),     # orange    |
            (255, 34, 22)         # red (highest bound)
        ]
        # Divide each color value by 255
        normalized_colors = [(r/255, g/255, b/255) for r, g, b in custom_colors]
        custom_cmap = ListedColormap(normalized_colors)
        bars = ax.bar(df['Wind Direction'], df['Shear_Rate'], normed=True, opening=1.0, bins=bins, cmap=custom_cmap, nsector=num_directions) # Plot the windrose plot based on the precentage and wind direction. 

        ax.set_radii_angle(angle=120)  # Adjust angle if necessary
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))  # Formatting as percentage

        if not show_cardinal: # False to show the cardinal directions
            ax.set_xticklabels([])

        if show_colorbar: # True to show the colorbar
            cmap = custom_cmap
            boundaries = bins
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True) # Set up the colorbar
            ax_cbar = fig.add_axes([0.1, 0.15, 0.8, 0.02])  # Position of the colorbar
            cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=boundaries, ticks=bins, spacing='uniform', orientation='horizontal') # Position the colorbar with uniform spacing and horizontal orientation
            cbar.set_label(r'Shear Stress [$\frac{N}{m^2}$]', fontsize=10) # Font size of the colorbar label
            cbar.ax.xaxis.set_tick_params(labelsize=9) # Adjust the fontsize of the tick labels

            # Use scientific notation for the tick labels
            cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cbar.ax.xaxis.get_offset_text().set_fontsize(9)  # Adjust the fontsize of the offset text

            # Ensure the labels use scientific notation
            cbar.ax.xaxis.get_major_formatter().set_powerlimits((0, 0)) # Ensure the labels use scientific notation

        # Generate filename based on ls_range
        filename = f'shear_{ls_range[0]}_{ls_range[1]}.svg' # Plot name based on ls_range
        plt.tight_layout() # Tight layout to avoid overlapping labels
        full_path = os.path.join(save_directory, filename)
        plt.savefig(full_path, format='svg')  # Save the figure as SVG
        #plt.show()
    else:
        print("Required columns (Um, Vm, shear_rate) missing in the dataframe")

folder_path = ['/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/East_Hellespontus',"/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/SITE_2"] # The path to the CSV files that contain the plots of interest
save_folder = ["/home/pruthvi/Desktop/HELLAS_SIMS/CODE/East_Hellespontus","/home/pruthvi/Desktop/HELLAS_SIMS/CODE/SITE_2"] # Path to where the plots will be saved

# List of (start_ls, end_ls) tuples
ls_ranges = [(0, 90), (91, 180), (181, 270), (271, 360), (286.4, 348.9), (348.9, 11), (11, 208.2), (208.2, 269.4), (269.4, 311.9), (311.9, 7.7)] # List of (start_ls, end_ls) tuples

for ii in range(len(folder_path)): # Loop through the folder paths
    max_shear, min_shear = find_max_min_shear_rate(folder_path[ii]) # Calculate the maximum and minimum shear rate for all of the CSV files in the directory
    print(max_shear, min_shear) # Printing for debugging purposes
    for start_ls, end_ls in ls_ranges: # Loop through the ls ranges
        show_colorbar = (start_ls, end_ls) in [(0, 90), (286.4, 348.9)] # Only show the colorbar for these two Ls ranges
        show_cardinal = not show_colorbar # If the colorbar is not shown, show the cardinal directions
        dfs = [] # Empty list to store the dataframes for each Ls range
        valid_files = [] # Empty list to store the file names for each Ls range
        for i in range(0, 360, 10): # Loop through the wind directions
            ls_value = i
            if start_ls <= end_ls:
                in_range = start_ls <= ls_value <= end_ls
            else:
                in_range = ls_value >= start_ls or ls_value <= end_ls
            if in_range:
                file_name = f"LS_{i}.nc.csv" # File name based on the Ls range
                full_path = os.path.join(folder_path[ii], file_name) # Full path to the CSV file
                if os.path.exists(full_path):
                    df = pd.read_csv(full_path) # Read the CSV file into a dataframe
                    dfs.append(df) # Add dataframe to the list of dataframes
                    valid_files.append(file_name) # Add file name to the list of file names

        if dfs:
            concatenated_df = pd.concat(dfs, ignore_index=True) # Concatenate the dataframes
            process_combined_dataframe(concatenated_df, valid_files, show_colorbar, show_cardinal, (start_ls, end_ls),save_directory=save_folder[ii]) # Create the combined plot for the Ls range
        else:
            print(f"No data files found in the specified Ls range: {start_ls}-{end_ls}.") # Error message if no valid data found in the CSV files