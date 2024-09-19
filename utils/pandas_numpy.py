import pandas as pd
import numpy as  np

# Function to convert each storm's data into a numpy array with the required format
def convert_storm_df_to_numpy_array(storm_group):
    storm_group = storm_group.sort_values(by='time')
    # Stack lat, lon, lat_grad, lon_grad, lat_grad2, lon_grad2 into a 2D array
    storm_array = np.stack([
        storm_group['lat'].values,
        storm_group['lon'].values,
        storm_group['lat_grad'].values,
        storm_group['lon_grad'].values,
        storm_group['lat_grad2'].values,
        storm_group['lon_grad2'].values
    ], axis=-1)
    return storm_array