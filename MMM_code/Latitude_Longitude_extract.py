import numpy as np
from ppclass import pp
import os
import netCDF4 as nc
import re
import pandas as pd

############################################################################################################
# This script extracts the Um, Vm, surface temperature and pressure values from the MMM netCDF output files
# Using the density values from the MDC and the fricition velocity values (UTSM), the script calculates the shear 
#############################################################################################################

def get_ls_from_time(rtf_file, search_time): # Function to get the YYYY:MM:DD_hh:mm:ss from from the rtf file that contain the conversion time
    search_date = search_time.split('_')[0]
    with open(rtf_file, 'r') as file:
        content = file.read()
        raw_text = re.sub(r"\\[a-z]+\d* ?|\\'[\d\w]{2}|[\{\}]|\\par", '', content) # Remove unwanted characters
        lines = raw_text.splitlines()
        for line in lines:
            if search_date in line:
                parts = line.split()
                if len(parts) > 2:
                    return parts[2]
    return None

def get_time_part(s): # Function to get the hh:mm:ss from the YYYY:MM:DD_hh:mm:ss
    return s.split('_')[-1]

# Read the density values from CSV
density_df = pd.read_csv("Density_values.csv") # Read the density values from CSV

folder_path = '/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES' # Specify the folder path for the netCDF files
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".nc")] # List all.nc files in the folder

for filename in csv_files: # Loop over all.nc files in the folder
    print filename

    ls_value_from_filename = int(filename.split('_')[1].split('.')[0]) # Extract the Ls value from the filename

    # Check if the extracted Ls value is present in the density_df
    if ls_value_from_filename in density_df['Ls'].values: # If the Ls value is present, get the corresponding density value
        
        density = density_df[density_df['Ls'] == ls_value_from_filename]['density (kg/m^3'].iloc[0]
    else:
        closest_ls_index = (density_df['Ls'] - ls_value_from_filename).abs().idxmin() # Find the closest Ls value in the density_df
        closest_ls_value = density_df.loc[closest_ls_index, 'Ls']
        density = density_df.loc[closest_ls_index, 'density (kg/m^3']
        print "No exact density value found for {}, using the closest value {} with density {}".format(ls_value_from_filename, closest_ls_value, density) # Print for debugging purposes

    nc_file = nc.Dataset(os.path.join(folder_path, filename), "r") # Open the NetCDF file
    times_array = nc_file.variables['Times'][:] # Get the time array from the NetCDF file
    times = []

    for time_entry in times_array: # loop over the time entries in the NetCDF file to format the time into a usable format
        if np.ma.is_masked(time_entry):
            continue
        time_string = ''.join(time_entry.tolist())
        times.append(time_string) 


    start_t = 47 # Start time
    end_t = 97 # End time
    # Certain runs stop at different times, so we need to set a if statement accordingly
    if filename == "LS_250.nc":
        end_t = 91
    elif filename == "LS_50.nc" or filename == 'LS_218.nc':
        end_t = 96
    elif filename == "LS_10.nc":
        end_t = 89

    data = []
    target_lat = -41.5112 # -41.4961 
    target_long = 45.55 # 44.6137
    for t in range(start_t, end_t): # Using Planetoplot to loop over all the times in the NetCDF file to extract specific values
        press = pp(file=os.path.join(folder_path, filename), var='PSFC', t=t, z=0.02, x=target_long, y=target_lat).getf() # x and y are the longitude and latitude of the target location
        temp = pp(file=os.path.join(folder_path, filename), var='tk', t=t, z=0.02, x=target_long, y=target_lat).getf() # surface temperature
        Um = pp(file=os.path.join(folder_path, filename), var='Um', t=t, z=0.02, x=target_long, y=target_lat).getf() # u-component of the velocity
        Vm = pp(file=os.path.join(folder_path, filename), var='Vm', t=t, z=0.02, x=target_long, y=target_lat).getf() # v-component of the velocity
        USTM= pp(file=os.path.join(folder_path, filename), var='USTM', t=t, z=0.02, x=target_long, y=target_lat).getf() # friction velocity
        shear_rate = (USTM**2) * density # Calculate the shear rate from the friction velocity and the density
        current_time = times[t]
        current_hhmmss = get_time_part(current_time)
        ls_value = get_ls_from_time("LMD_CALENDAR.rtf", current_time)
        # Add the extracted values to the data list
        data.append({'surface_temperature': temp, 'surface_pressure': press, 'time_hhmmss': current_hhmmss, 'Um': Um, 'Vm': Vm, 'shear_rate': shear_rate})

    df = pd.DataFrame(data) # Convert the data list to a pandas dataframe
    df.to_csv(os.path.join(folder_path, str(filename) + '.csv'), index=False) # Save the pandas dataframe to a CSV file
    
