import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    np.fill_diagonal(distance_matrix.values, 0)
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix
df = pd.read_csv('datasets/dataset-2.csv')
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)

def unroll_distance_matrix(distance_matrix) -> pd.DataFrame:
    """
    Unroll the distance matrix into a DataFrame with columns id_start, id_end, and distance.

    Args:
        distance_matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled distance DataFrame
    """
    unrolled_data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                unrolled_data.append([id_start, id_end, distance_matrix.at[id_start, id_end]])
    return pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
unrolled_distance_df = unroll_distance_matrix(distance_matrix)
print(unrolled_distance_df)

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
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1
    avg_distances = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = avg_distances[(avg_distances >= lower_threshold) & (avg_distances <= upper_threshold)].index.tolist()
    
    return sorted(ids_within_threshold)
reference_id = 1001402
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_id)
print(ids_within_threshold)

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df
    
toll_rate_df = calculate_toll_rate(unrolled_distance_df)
print(toll_rate_df)


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekday_discounts = [
        (time(0, 0), time(10, 0), 0.8),
        (time(10, 0), time(18, 0), 1.2),
        (time(18, 0), time(23, 59, 59), 0.8)
    ]
    weekend_discount = 0.7
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    new_rows = []
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        for day in days_of_week:
            if day in ['Saturday', 'Sunday']:
                for hour in range(24):
                    start_time = time(hour, 0)
                    end_time = time(hour, 59, 59)
                    new_row = {
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': distance * 0.8 * weekend_discount,
                        'car': distance * 1.2 * weekend_discount,
                        'rv': distance * 1.5 * weekend_discount,
                        'bus': distance * 2.2 * weekend_discount,
                        'truck': distance * 3.6 * weekend_discount
                    }
                    new_rows.append(new_row)
            else:
                for start_time, end_time, discount in weekday_discounts:
                    new_row = {
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': distance * 0.8 * discount,
                        'car': distance * 1.2 * discount,
                        'rv': distance * 1.5 * discount,
                        'bus': distance * 2.2 * discount,
                        'truck': distance * 3.6 * discount
                    }
                    new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    return new_df
time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_rate_df)

