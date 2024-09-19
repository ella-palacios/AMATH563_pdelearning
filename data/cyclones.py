# Uncomment and install if needed
# !pip install netCDF4 matplotlib cartopy numpy autograd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import interp1d
import sys
sys.path.append('..')

def extract_cyclones_data(file_path, print_all_vars=False):
    # Open the NetCDF file
    dataset = Dataset(file_path, 'r')

    # Print out all the variables available in the NetCDF file
    print("Variables in the NetCDF file:")
    if print_all_vars == True:
        for var_name in dataset.variables:
            print(var_name)

    # Extract relevant data
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]
    storm_ids = dataset.variables['sid'][:]
    time = dataset.variables['time'][:]  
    storm_ids = [''.join(sid.astype(str)).strip() for sid in storm_ids]  # Convert to string format
    unique_storm_ids = np.unique(storm_ids)
    num_storms = len(unique_storm_ids)
    print("Examining", num_storms, "storms")

    valid_storm_ids = []
    for unique_storm_id in unique_storm_ids:
        storm_indices = [i for i, sid in enumerate(storm_ids) if sid == unique_storm_id]
        storm_lat = lat[storm_indices, :].flatten()
        storm_lat = np.array([float(x) for x in storm_lat])
        storm_lat = storm_lat[~np.isnan(storm_lat)]
        storm_lon = lon[storm_indices, :].flatten()
        storm_lon = np.array([float(x) for x in storm_lon])
        storm_lon = storm_lon[~np.isnan(storm_lon)]
        if len(storm_lat) != len(storm_lon):
            print("error mismatch of dimensions", len(storm_lat), len(storm_lon))
        elif (len(storm_lat) < 10):
            print("truncating storm", unique_storm_id, "due to lack of time steps")
        else:
            valid_storm_ids.append(unique_storm_id)

    # Filter the original data
    valid_storm_indices = [i for i, sid in enumerate(storm_ids) if sid in valid_storm_ids]
    lat = lat[valid_storm_indices, :]
    lon = lon[valid_storm_indices, :]
    time = time[valid_storm_indices, :]
    print("Filtered to", len(valid_storm_ids), "storms")

    # Close the NetCDF file
    dataset.close()

    return valid_storm_ids, time, lat, lon

def make_cyclones_df(valid_storm_ids, time, lat, lon):
    # Combine the data into a pandas DataFrame for easier handling
    data = []
    for i, storm_id in enumerate(valid_storm_ids):
        for j in range(lat.shape[1]):
            if np.isfinite(lat[i, j]) and np.isfinite(lon[i, j]):
                data.append([storm_id, time[i, j], lat[i, j], lon[i, j]])

    df = pd.DataFrame(data, columns=['storm_id', 'time', 'lat', 'lon'])
    unique_storm_count = df['storm_id'].nunique()
    print(unique_storm_count)

    # Sort by storm_id and time
    df = df.sort_values(by=['storm_id', 'time'])
    df['time'] = df.groupby('storm_id')['time'].transform(lambda x: x - x.min())


    # Interpolate missing time steps if any
    interpolated_data = []
    for storm_id in valid_storm_ids:
        storm_data = df[df['storm_id'] == storm_id]
        if len(storm_data) > 1:
            try:
                f_lat = interp1d(storm_data['time'], storm_data['lat'], kind='linear', fill_value='extrapolate')
                f_lon = interp1d(storm_data['time'], storm_data['lon'], kind='linear', fill_value='extrapolate')
                time_storm = storm_data['time'].values
                lat_interp = f_lat(time_storm)
                lon_interp = f_lon(time_storm)
                
                # Ensure no NaNs in the interpolated data
                if np.any(np.isnan(lat_interp)) or np.any(np.isnan(lon_interp)):
                    print(f"Dropping storm {storm_id} due to NaNs in interpolated data")
                    continue  # Skip this storm if NaNs are found
                
                interpolated_data.extend(zip([storm_id]*len(time_storm), time_storm, lat_interp, lon_interp))
            except Exception as e:
                print(f"Error during interpolation for storm {storm_id}: {e}")
                continue
        else:
            interpolated_data.extend(storm_data.values)
    df_interp = pd.DataFrame(interpolated_data, columns=['storm_id', 'time', 'lat', 'lon'])
    return df_interp

def plot_trajectories(valid_storm_ids, lat, lon):# Set up the map with Cartopy
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.gridlines()

    # Plot the paths of the tropical cyclones
    for storm_id in valid_storm_ids:
        storm_indices = [i for i, sid in enumerate(valid_storm_ids) if sid == storm_id]
        storm_lat = lat[storm_indices, :].flatten()
        storm_lon = lon[storm_indices, :].flatten()
        
        # Remove missing data points (usually marked with very high or low values)
        valid_mask = np.isfinite(storm_lat) & np.isfinite(storm_lon)
        storm_lat = storm_lat[valid_mask]
        storm_lon = storm_lon[valid_mask]

        # Plot the line for the storm
        ax.plot(storm_lon, storm_lat, transform=ccrs.PlateCarree(), linewidth=1)

    plt.title('Tropical Cyclone Paths from IBTrACS Dataset')
    plt.show()

# Compute the gradients using finite differences and extend the df with gradients columns
def compute_gradients(df):
    df = df.sort_values(by=['storm_id', 'time'])
    df['lat_grad'] = df.groupby('storm_id')['lat'].diff().fillna(0)
    df['lon_grad'] = df.groupby('storm_id')['lon'].diff().fillna(0)
    df['lat_grad2'] = df.groupby('storm_id')['lat_grad'].diff().fillna(0)
    df['lon_grad2'] = df.groupby('storm_id')['lon_grad'].diff().fillna(0)
    return df