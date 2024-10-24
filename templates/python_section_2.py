import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    
     # Convert the DataFrame to a NumPy array for easier manipulation
    distance_matrix = df.to_numpy()

    # Number of toll locations (n x n matrix)
    num_locations = distance_matrix.shape[0]

    # Apply the Floyd-Warshall algorithm to find the shortest path between all pairs
    for k in range(num_locations):
        for i in range(num_locations):
            for j in range(num_locations):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    # Ensure the diagonal remains 0 (distance from a location to itself)
    np.fill_diagonal(distance_matrix, 0)

    # Convert the NumPy array back to a DataFrame
    df = pd.DataFrame(distance_matrix, index=df.index, columns=df.columns)

    return df



def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    
 # List to hold the unrolled data
    unrolled_data = []

    # Iterate through the matrix rows and columns
    for id_start in df.index:
        for id_end in df.columns:
            # Avoid diagonal elements (id_start == id_end)
            if id_start != id_end:
                # Append to the list
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': df.loc[id_start, id_end]
                })

    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    
 # Calculate the average distance for the reference_value
    ref_avg_distance = df[df['id_start'] == reference_value]['distance'].mean()

    # Define the 10% threshold bounds
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    # Find unique id_start values
    id_start_values = df['id_start'].unique()

    # List to store id_start values within the 10% threshold
    within_threshold = []

    # For each id_start value, calculate its average distance
    for id_start in id_start_values:
        avg_distance = df[df['id_start'] == id_start]['distance'].mean()
        
        # Check if the average distance lies within the 10% threshold
        if lower_bound <= avg_distance <= upper_bound:
            within_threshold.append(id_start)

    # Return the sorted list of id_start values
    return sorted(within_threshold)
    


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

     # Define rate coefficients
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add them as new columns
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']

    return df



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    
    from datetime import time

     # Convert start_time and end_time columns to datetime.time objects
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time

    # Define discount factors
    weekday_discounts = {
        '00:00:00-10:00:00': 0.8,
        '10:00:00-18:00:00': 1.2,
        '18:00:00-23:59:59': 0.8
    }
    weekend_discount = 0.7

    # Define time ranges
    time_ranges = {
        '00:00:00-10:00:00': (time(0, 0, 0), time(10, 0, 0)),
        '10:00:00-18:00:00': (time(10, 0, 0), time(18, 0, 0)),
        '18:00:00-23:59:59': (time(18, 0, 0), time(23, 59, 59))
    }

    # Function to apply discount based on time ranges and days
    def apply_discount(row):
        is_weekend = row['start_day'] in ['Saturday', 'Sunday']
        if is_weekend:
            discount = weekend_discount
        else:
            for key, (start, end) in time_ranges.items():
                if start <= row['start_time'] <= end:
                    discount = weekday_discounts[key]
                    break
        # Apply the discount to all vehicle columns
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] = row[vehicle] * discount
        return row

    # Apply the function to each row
    df = df.apply(apply_discount, axis=1)
    
    return df
