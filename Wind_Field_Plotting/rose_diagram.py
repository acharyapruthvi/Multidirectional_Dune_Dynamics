import re
import os
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import math
from datetime import datetime
import seaborn as sns
from windrose import WindroseAxes
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams.update({'font.size': 10})  

def find_max_min_windspeed(folder_path):
    """
    Reads all CSV files in the specified folder, extracts the 'shear_rate' column,
    and returns the maximum and minimum shear rates.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        tuple: A tuple containing the maximum and minimum shear rates.
    """
    try:
        max_wind_speed = float('-inf')
        min_wind_speed= float('inf')

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                df['Wind_Speed'] = calculate_wind_speed(df['Um'], df['Vm']).round(0)
                if 'Wind_Speed' in df.columns:
                    max_wind_speed = max(max_wind_speed, df['Wind_Speed'].max())
                    min_wind_speed = min(min_wind_speed, df['Wind_Speed'].min())

        if max_wind_speed != float('-inf') and min_wind_speed != float('inf'):
            return max_wind_speed, min_wind_speed
        else:
            return None, None  # No valid data found in the CSV files
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return None, None

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
        
        num_directions = 30
        df["bin"] = pd.cut(df["Wind Direction"], bins=num_directions, labels=range(num_directions))
        sums = df.groupby("bin")["Wind_Speed"].sum()
        sums_percentage = (sums / sums.sum()) * 100
        
        fig = plt.figure(figsize=(1.2*2.125, 1.2*2.7))
        ax = WindroseAxes.from_ax(fig=fig)
        bins = np.linspace(0, max_speed, 6)
        bins = [ round(elem, 1) for elem in bins ]
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

        bars = ax.bar(df['Wind Direction'], df['Wind_Speed'], normed=True, opening=1.0, bins=bins, cmap=custom_cmap, nsector=num_directions)
        
        # Modify concentric rings to show percentage
        ax.set_radii_angle(angle=120)  # Adjust angle if necessary
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))  # Formatting as percentage

        if not show_cardinal:
            ax.set_xticklabels([])  # Hide cardinal direction labels

        if show_colorbar:
            cmap = custom_cmap
            boundaries = bins
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
            ax_cbar = fig.add_axes([0.1, 0.15, 0.8, 0.02]) # Modify these values as needed
            cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=boundaries, ticks=bins, spacing='proportional', orientation='horizontal')
            cbar.set_label(r'Wind Speed [$\frac{m}{s}$]', fontsize=10)  # Setting font size for the colorbar's label
            cbar.ax.xaxis.set_tick_params(labelsize=10)  # Setting font size for the colorbar's tick labels
        # Generate filename based on ls_range
        filename = f'windspeed_{ls_range[0]}_{ls_range[1]}.svg'
        plt.tight_layout()
        full_path = os.path.join(save_directory, filename) 
        plt.savefig(full_path, format='svg')  # Save the figure as SVG
        #plt.show()
    else:
        print("Required columns (Um, Vm, shear_rate) missing in the dataframe")

folder_path = ['/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/East_Hellespontus',"/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/SITE_2"] # The path to the CSV files that contain the plots of interest
save_folder = ["/home/pruthvi/Desktop/HELLAS_SIMS/CODE/East_Hellespontus","/home/pruthvi/Desktop/HELLAS_SIMS/CODE/SITE_2"] # Path to where the plots will be saved

# List of (start_ls, end_ls) tuples
ls_ranges = [(0, 90), (91, 180), (181, 270), (271, 360), (286.4, 348.9), (348.9, 11), (11, 208.2), (208.2, 269.4), (269.4, 311.9), (311.9, 7.7)]

for ii in range(len(folder_path)):
    max_speed, min_speed = find_max_min_windspeed(folder_path[ii]) # Calculate the maximum and minimum shear rate for all of the CSV files in the directory
    print(max_speed, min_speed) # Printing for debugging purposes
    for start_ls, end_ls in ls_ranges:
        show_colorbar = (start_ls, end_ls) in [(0, 90), (286.4, 348.9)]
        show_cardinal = not show_colorbar
        dfs = []
        valid_files = []
        for i in range(0, 360, 10):
            ls_value = i
            if start_ls <= end_ls:
                in_range = start_ls <= ls_value <= end_ls
            else:
                in_range = ls_value >= start_ls or ls_value <= end_ls
            if in_range:
                file_name = f"LS_{i}.nc.csv"
                full_path = os.path.join(folder_path[ii], file_name)
                if os.path.exists(full_path):
                    df = pd.read_csv(full_path)
                    dfs.append(df)
                    valid_files.append(file_name)

        if dfs:
            concatenated_df = pd.concat(dfs, ignore_index=True)
            process_combined_dataframe(concatenated_df, valid_files, show_colorbar, show_cardinal, (start_ls, end_ls),save_directory=save_folder[ii])
        else:
            print(f"No data files found in the specified Ls range: {start_ls}-{end_ls}.")
