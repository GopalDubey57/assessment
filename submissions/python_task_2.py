import pandas as pd
import numpy as np
import warnings
from datetime import time
warnings.filterwarnings("ignore")
df_real = pd.read_csv('../datasets/dataset-3.csv')
df = df_real.copy()

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    # Initialize the matrix with 0 values
    distance_matrix = distance_matrix.fillna(0)

    # Update the matrix with cumulative distances along known routes
    for index, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] += distance
        distance_matrix.at[id_end, id_start] += distance  # Ensure symmetry

    return distance_matrix


q1 = calculate_distance_matrix(df)
# print(q1)

def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(np.bool))
    upper_triangle = upper_triangle.stack().reset_index()
    upper_triangle.columns = ['id_start', 'id_end', 'distance']

    # Create a DataFrame from the lower triangle of the matrix
    lower_triangle = distance_matrix.where(np.tril(np.ones(distance_matrix.shape), k=-1).astype(np.bool))
    lower_triangle = lower_triangle.stack().reset_index()
    lower_triangle.columns = ['id_end', 'id_start', 'distance']

    # Concatenate the upper and lower triangles to get all unique combinations
    result_df = pd.concat([upper_triangle, lower_triangle], ignore_index=True)

    # Filter out rows where 'id_start' is equal to 'id_end'
    result_df = result_df[result_df['id_start'] != result_df['id_end']]

    return result_df

q2 = unroll_distance_matrix(q1)
# print(q2)


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
    reference_df = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the lower and upper thresholds within 10%
    lower_threshold = 0.9 * average_distance
    upper_threshold = 1.1 * average_distance

    # Filter DataFrame for values within the threshold and sort the unique IDs
    result_ids = (
        df[(df['distance'].between(lower_threshold, upper_threshold)) & (df['id_start'] != df['id_end'])]
        .sort_values('id_start')
        ['id_start']
        .unique()
        .tolist()
    )

    return result_ids



q3 = find_ids_within_ten_percentage_threshold(q2,100)
print(q3)
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create a copy of the input DataFrame to avoid modifying the original
    df_with_rates = df.copy()

    # Create new columns for each vehicle type with their respective toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df_with_rates[vehicle_type] = df_with_rates['distance'] * rate_coefficient

    return df_with_rates.drop('distance',axis=1)


q4 = calculate_toll_rate(q2)
# print(q4)

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0)),
        (time(10, 0, 0), time(18, 0, 0)),
        (time(18, 0, 0), time(23, 59, 59))
    ]

    weekend_time_ranges = [(time(0, 0, 0), time(23, 59, 59))]

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(df)

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_time'] = df['end_day'] = df['end_time'] = None

    # Iterate over unique (id_start, id_end) pairs
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        # Create a template DataFrame for the 24-hour period and 7 days of the week
        template_df = pd.DataFrame(index=pd.MultiIndex.from_product(
            [range(7), pd.date_range('00:00:00', '23:59:59', freq='1S')],
            names=['day', 'time']
        ))

        # Initialize vehicle columns with zeros
        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            template_df[vehicle_type] = 0

        # Populate start_day, start_time, end_day, and end_time
        template_df['start_day'] = template_df.index.get_level_values('day').map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        template_df['start_time'] = template_df.index.get_level_values('time').time

        template_df['end_day'] = template_df['start_day'].shift(-1)
        template_df['end_time'] = template_df['start_time'].shift(-1)

        template_df.loc[template_df.index[-1], 'end_day'] = template_df['start_day'].iloc[0]
        template_df.loc[template_df.index[-1], 'end_time'] = template_df['start_time'].iloc[0]

        # Apply discount factors based on time ranges and weekdays/weekends
        for i, time_range in enumerate(weekday_time_ranges if template_df['start_day'].iloc[0] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else weekend_time_ranges):
            mask = (template_df['start_time'] >= time_range[0]) & (template_df['start_time'] <= time_range[1])
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                template_df.loc[mask, vehicle_type] *= (weekday_discount_factors[i] if template_df['start_day'].iloc[0] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else weekend_discount_factor)

        # Merge the template DataFrame with the original DataFrame
        merged_df = pd.merge(template_df.reset_index(), group, how='left', on=['day', 'time'])

        # Fill NaN values with 0 for the vehicle columns
        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            merged_df[vehicle_type] = merged_df[vehicle_type].fillna(0)

        # Update the original DataFrame with the calculated values
        df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), ['start_day', 'start_time', 'end_day', 'end_time']] = merged_df[['start_day', 'start_time', 'end_day', 'end_time']].values[0]

        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), vehicle_type] = merged_df[vehicle_type]

    return df
q5  = calculate_time_based_toll_rates(q3)
print(q5)