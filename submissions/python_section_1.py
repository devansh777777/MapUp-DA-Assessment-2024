from typing import Dict, List
import re
import pandas as pd
import polyline
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    for i in range(0, len(lst), n):
        group = lst[i:i + n]
        for j in range(len(group)):
            lst[i + j] = group[len(group) - 1 - j]
    return lst

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(current_dict, parent_key=''):
        flattened = {}
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                flattened.update(flatten(value, new_key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    indexed_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        flattened.update(flatten(item, indexed_key))
                    else:
                        flattened[indexed_key] = item
            else:
                flattened[new_key] = value

        return flattened
    result = flatten(nested_dict)
    return dict(result)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, available):
        if not available:
            result.append(path)
            return
        prev = None
        for i in range(len(available)):
            if available[i] == prev:
                continue
            backtrack(path + [available[i]], available[:i] + available[i+1:])
            prev = available[i]

    nums.sort() 
    result = []
    backtrack([], nums)
    return result
    pass

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    dates = re.findall(date_pattern, text)
    return dates
    pass

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    R = 6371000  
    for i in range(1, len(df)):
        lat1, lon1 = radians(df.loc[i - 1, 'latitude']), radians(df.loc[i - 1, 'longitude'])
        lat2, lon2 = radians(df.loc[i, 'latitude']), radians(df.loc[i, 'longitude'])
        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        df.loc[i, 'distance'] = R * c
    return pd.DataFrame(df)

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    return final_matrix

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    def get_day_index(day_str):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days.index(day_str)

    def generate_week_time_slots():
        week_slots = []
        for day in range(7):
            for hour in range(24):
                week_slots.append((day, hour))
        return set(week_slots)

    def extract_time_slots_optimized(row):
        start_day_idx = get_day_index(row['startDay'])
        end_day_idx = get_day_index(row['endDay'])
        start_time = datetime.strptime(row['startTime'], "%H:%M:%S").hour
        end_time = datetime.strptime(row['endTime'], "%H:%M:%S").hour
        slots = set()
        current_day = start_day_idx
        while True:
            if current_day == start_day_idx: 
                for hour in range(start_time, 24):
                    slots.add((current_day, hour))
            else:
                for hour in range(0, 24):
                    slots.add((current_day, hour))

            if current_day == end_day_idx:
                for hour in range(0, end_time + 1):
                    slots.add((current_day, hour))
                break

            current_day = (current_day + 1) % 7

        return slots
    grouped = df.groupby(['id', 'id_2'])
    complete_week_slots = generate_week_time_slots()
    incorrect_flags = []
    for (id_val, id_2_val), group in grouped:
        total_slots = set()
        for _, row in group.iterrows():
            total_slots.update(extract_time_slots_optimized(row))
        incorrect_flags.append(total_slots != complete_week_slots)
    index = pd.MultiIndex.from_tuples(grouped.groups.keys(), names=["id", "id_2"])
    return pd.Series(incorrect_flags, index=index)
df = pd.read_csv("datasets/dataset-1.csv")
result = time_check(df)
print(result)
