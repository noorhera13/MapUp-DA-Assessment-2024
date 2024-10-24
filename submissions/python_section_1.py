from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    rev_lst = []
    for i in range(len(lst)-1,-1,-1):
        rev_lst.append(lst[i])
    return rev_lst
    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    res = {}
    for a in lst:
        l = len(a)
        if l not in res:
            res[l] = []
            res[l].append(a)
    res = res.sorted(res.items())
    
    return res

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here

    def _flatten(current_dict, parent_key=''):
        items = []
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, val in enumerate(v):
                    items.extend(_flatten(val, f"{new_key}[{i}]", sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten(nested_dict)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(path, used, result):
        if len(path) == len(nums):
            result.append(path[:])  # Make a copy of path and add to result
            return
        for i in range(len(nums)):
            # Skip used elements and duplicates
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used, result)
            path.pop()  # Backtrack: remove the last element and mark it unused
            used[i] = False

    nums.sort()  # Sort to handle duplicates more easily
    result = []
    used = [False] * len(nums)  # Track if an element is already used
    backtrack([], used, result)
    
    return result



import re

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    pattern = r"\b(\d{2}-\d{2}-\d{4})|(\d{2}/\d{2}/\d{4})|(\d{4}\.\d{2}\.\d{2})\b"
    
    # Find all matches using the pattern
    matches = re.findall(pattern, text)
    
    # Extract the valid date from the matched groups
    dates = []
    for match in matches:
        for date in match:
            if date:
                dates.append(date)
    
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


import polyline
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    from math import radians, sin, cos, sqrt, atan2
    from typing import Tuple

    def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000  
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

def decode_polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Decodes a polyline string into a DataFrame with columns for latitude, longitude, and distance between points.
    
    Parameters:
    polyline_str (str): A polyline encoded string.
    
    Returns:
    pd.DataFrame: A DataFrame with 'latitude', 'longitude', and 'distance' columns.
    """
    
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)
    
    # Initialize list to hold distance values (first point has distance 0)
    distances = [0]  # First point has no previous point, so distance is 0
    
    # Calculate the distance between each successive coordinate
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i-1]
        lat2, lon2 = coordinates[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    # Create DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = distances
    
    return df



def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotates a given square matrix by 90 degrees clockwise and replaces each element
    with the sum of all elements in the same row and column, excluding itself.
    
    Parameters:
    matrix (List[List[int]]): A 2D list representing the square matrix to be transformed.
    
    Returns:
    List[List[int]]: A new matrix with transformed values after rotation.
    """
    
    n = len(matrix)
    
    #1 Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    #2 Create the transformed matrix
    transformed_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of all elements in the same row and column, excluding the element itself
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix

# Example usage
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_transform_matrix(matrix)
for row in result:
    print(row)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

     # Create a multi-index based on `id` and `id_2`
    df.set_index(['id', 'id_2'], inplace=True)
    
    # Convert the days of the week to numeric for easier comparison (assuming 1 = Monday, 7 = Sunday)
    days_of_week = range(1, 8)
    full_day_start = pd.Timestamp('00:00:00')
    full_day_end = pd.Timestamp('23:59:59')

    # Create a helper function to check completeness for each unique (id, id_2) pair
    def check_completeness(group):
        # Extract the start and end days and times
        start_days = group['startDay'].unique()
        end_days = group['endDay'].unique()
        
        # Check if all 7 days are covered
        days_covered = set(days_of_week).issubset(set(start_days)) and set(days_of_week).issubset(set(end_days))
        
        # Check if each day has full 24-hour coverage
        time_covered = all(group['startTime'].min() == full_day_start) and all(group['endTime'].max() == full_day_end)
        
        return days_covered and time_covered
    
    # Apply the completeness check to each group (grouped by `id` and `id_2`)
    completeness_check = df.groupby(['id', 'id_2']).apply(check_completeness)

    # Return the boolean series indicating whether each (id, id_2) has incomplete timestamps
     return ~completeness_check

