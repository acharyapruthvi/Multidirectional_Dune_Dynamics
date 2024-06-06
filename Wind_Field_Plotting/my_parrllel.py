import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
from ppclass import pp
import netCDF4 as nc

################################################################################################################################
# This script reads the MMM netCDF files and plots the topography and wind field over the surface temperauture. 
# The plot is centered at the East Hellespontus site (White Dot) and a site 40 km from the East Hellespontus site (Black Dot)
################################################################################################################################

def max_min_temp(nc_file, start_t, end_t, tar_lat, tar_lon,crop_size):
    """
    Reads a NetCDF file containing surface temperature data and returns the maximum and minimum surface temperatures
    within the specified time range.

    Args:
        nc_file (nc file with all paramters): Path to the NetCDF file.
        t_start (float): Start time (in seconds) for the desired range.
        t_end (float): End time (in seconds) for the desired range.
        tar_lat (float): Target latitude of choice
        tar_long (float) : Target longitude of choice
        crop_size (int): Crop size you want

    Returns:
        tuple: A tuple containing the maximum and minimum surface temperatures.
    """
    max_temps = []
    min_temps = []
    for t in range(start_t, end_t):
        var = pp(file=filename, var='tk', t=t, z=0.02).getf()
        lat = pp(file=filename, var='XLAT', t=t).getf()
        long = pp(file=filename, var='XLONG', t=t).getf()
        
        #FINDING THE TARGET COORDINATE
        dist = (lat - tar_lat)**2 + (long - tar_lon)**2
        index = np.unravel_index(np.argmin(dist), dist.shape)
        i_min, i_max = max(0, index[0] - crop_size), min(var.shape[0], index[0] + crop_size)
        j_min, j_max = max(0, index[1] - crop_size), min(var.shape[1], index[1] + crop_size)
        # Crop data arrays
        var_crop = var[i_min:i_max, j_min:j_max]
        var_crop = np.array(var_crop)

        max_temps.append(np.max(var_crop))
        min_temps.append(np.min(var_crop))

    return (max(max_temps), min(min_temps))

def plot_data_with_points(filename, t, output_dir, target_coords, colors):
    # Extract the numerical part from the filename assuming format "LS_##.nc"
    base_name = os.path.basename(filename)
    number_part = ''.join([char for char in base_name if char.isdigit()])

    # Define the path for saving the figure
    save_path = os.path.join(output_dir, 'LS_{}_{}.png'.format(number_part, t))

    # Check if the file already exists
    if os.path.exists(save_path):
        print("File {} already exists. Skipping...".format(save_path))
        return

    # Load data
    nc_file = nc.Dataset(filename, "r")
    top = pp(file=filename, var='HGT', t=t, z=0.02).getf()
    var = pp(file=filename, var='tk', t=t, z=0.02).getf()
    lat = pp(file=filename, var='XLAT', t=t).getf()
    long = pp(file=filename, var='XLONG', t=t).getf()
    Um = pp(file=filename, var='Um', t=t, z=0.02).getf()
    Vm = pp(file=filename, var='Vm', t=t, z=0.02).getf()

    # Find the closest grid point to the first target coordinate
    coord = target_coords[0]
    dist = (lat - coord[0])**2 + (long - coord[1])**2
    index = np.unravel_index(np.argmin(dist), dist.shape)

    # Define the crop size around the closest grid point
    crop_size = 10  # Half-size for 20x20 region
    i_min, i_max = max(0, index[0] - crop_size), min(var.shape[0], index[0] + crop_size)
    j_min, j_max = max(0, index[1] - crop_size), min(var.shape[1], index[1] + crop_size)

    # Crop data arrays
    var_crop = var[i_min:i_max, j_min:j_max]
    long_crop = long[i_min:i_max, j_min:j_max]
    lat_crop = lat[i_min:i_max, j_min:j_max]
    Um_crop = Um[i_min:i_max, j_min:j_max]
    Vm_crop = Vm[i_min:i_max, j_min:j_max]
    top_crop = top[i_min:i_max, j_min:j_max]

    # Create figure and axis for the cropped region
    fig, ax = plt.subplots(figsize=(8,8))

    # Plot temperature contour for the cropped region
    print round(min_temp,0), round(max_temp,0)
    levels = np.linspace(min_temp, max_temp, 21) 
    temp_contour = ax.contourf(long_crop, lat_crop, var_crop, levels=levels, cmap='coolwarm',vmin=int(min_temp), vmax=int(max_temp))
    plt.colorbar(temp_contour, ax=ax, label='Temperature (K)')

    # Plot wind vectors for the cropped region
    skip = 2
    Q = ax.quiver(long_crop[::skip, ::skip], lat_crop[::skip, ::skip], Um_crop[::skip, ::skip], Vm_crop[::skip, ::skip], scale=250)

    # Plot target points, retrieve pressures, and prepare legend entries
    legend_handles = []
    for coord, color in zip(target_coords, colors):
        point = ax.scatter(coord[1], coord[0], color=color, s=100, edgecolor='black', zorder=5)
        # Retrieve pressure at the location
        pressure_at_location = pp(file=filename, var='PSFC', z=0.02, t=t, x=coord[1], y=coord[0]).getf()
        label = 'Coord: ({:.2f}, {:.2f}), Pressure: {:.2f} Pa'.format(coord[0], coord[1], pressure_at_location)
        legend_handles.append((point, label))

    # HGT CONTOURS
    # Overlay the temperature contours
    min_var = np.min(top_crop)
    max_var = np.max(top_crop)
    levels = np.arange(min_var, max_var, 500)  # adjust the step as needed
    cs = plt.contour(long_crop, lat_crop, top_crop, levels, colors = 'black')

    plt.clabel(cs, cs.levels, inline=True, fmt='%d m', fontsize=10)  # Label contour lines

    # Add custom legend for coordinates and pressure
    ax.legend([handle for handle, label in legend_handles], [label for handle, label in legend_handles], loc='upper left')

    # Add reference quiver for wind speed
    ref_scale = 20  # Change this based on your data's typical wind speeds
    qk = ax.quiverkey(Q, 0.8, 0.90, ref_scale, '%d m/s' % ref_scale, labelpos='E', coordinates='figure')

    # Set the ticks to match the latitudes and longitudes
    ax.set_xticks(long_crop[0, :])  # Assuming evenly spaced longitudes
    ax.set_yticks(lat_crop[:, 0])  # Assuming evenly spaced latitudes
    ax.set_xlim([long_crop[0, 0], long_crop[0, -1]])  # Set x-axis limits
    ax.set_ylim([lat_crop[0, 0], lat_crop[-1, 0]])  # Set y-axis limits
    ax.set_xticklabels(['{:.2f}'.format(lng) for lng in long_crop[0, :]], rotation=90)

    ax.grid(True)  # Enable the grid

    # Set plot labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Cropped Wind and Temperature at time %s' % times[t][-8:])
    #plt.show()
    
    # Save the figure with the specific naming convention
    plt.savefig(save_path, format='png')
    plt.close(fig)

# Example usage
target_coords = [(-41.5112, 45.55), (-41.4961, 44.6137)]
colors = ['white', 'black']
folder_path = "/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES" # Where the netcdf files are located
output_dir = '/home/pruthvi/Desktop/HELLAS_SIMS/NC_FILES/test' # Where the figures will be saved
filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(".nc")]
for filename in filenames:
    
    nc_file = nc.Dataset(os.path.join(folder_path, filename), "r")
    times_array = nc_file.variables['Times'][:]
    times = []

    for time_entry in times_array:
        if np.ma.is_masked(time_entry):
            continue
        time_string = ''.join(time_entry.tolist())
        times.append(time_string)
    t_start = 48
    t_end = 98
    print(len(times))
    # Readjust the start and end times based on the number of output timesteps in the netcdf file
    if filename == 'LS_10.nc' or filename == 'LS_328.nc' or filename == 'LS_50.nc' or filename == 'LS_20.nc' or filename=='LS_218.nc' or filename == 'LS_250.nc' or filename=='LS_338.nc':
        t_end = len(times)

    full_path = os.path.join(folder_path, filename)
    max_temp, min_temp = max_min_temp(full_path,start_t=t_start, end_t=t_end, crop_size=10, tar_lat=-41.5112, tar_lon=45.55) # Get the max and min temperatures at the target coordinate site
    print filename , min_temp, max_temp
    for t in range(t_start, t_end): # Loop through the timesteps and plot the data
        print filename , t # Print for debugging purposes
        plot_data_with_points(full_path, t, output_dir, target_coords, colors)
