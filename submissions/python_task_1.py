import pandas as pd



df_real = pd.read_csv('../datasets/dataset-1.csv')
df = df_real.copy()
def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a df  for id combinations.

    Args:
        df (pandas.df)

    Returns:
        pandas.df: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.ṇ
    """
    # Write your logic here
    df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    return df

q1 = generate_car_matrix(df)

# print(df)

def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.df)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    
    df.loc[df['car'] <= 15, 'car_type'] = 'low'
    df.loc[(df['car'] > 15) & (df['car'] <= 25), 'car_type'] = 'medium'
    df.loc[df['car'] > 25, 'car_type'] = 'high'

    # Count occurrences of each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys

    return dict(sorted(type_counts.items()))

q2 = get_type_count(df)
# print(q2)

def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.df)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_mean = df['bus'].mean()
    
  
    bus_indices = df[df['bus'] > 2 * bus_mean].index
    
    
    return list(bus_indices.sort_values())

q3 = get_bus_indexes(df)
# print(q3)

def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.df)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_truck_means = df.groupby('route')['truck'].mean()


    selected_routes = route_truck_means[route_truck_means > 7].index
    

    return list(selected_routes.sort_values())

q4 = filter_routes(df)
# print(q4)

def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.df)

    Returns:
        pandas.df: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

q5 = multiply_matrix(q1)
# print(q5)

new_df = pd.read_csv('../datasets/dataset-2.csv')



def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.df)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    # Combine endDay and endTime columns to create an 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Create a MultiIndex with 'id' and 'id_2'
    multi_index = ['id', 'id_2']

    # Check if each (id, id_2) pair has correct timestamps
    completeness_check = df.groupby(multi_index).apply(
    lambda group: (
        (group['start_timestamp'].min() == group['start_timestamp'].dt.floor('D')) &
        (group['end_timestamp'].max() == group['end_timestamp'].dt.ceil('D')) &
        (group['start_timestamp'].dt.dayofweek.min() == 0) &
        (group['end_timestamp'].dt.dayofweek.max() == 6)
    ).all()  # Use .all() to check if all conditions are True
)


    return pd.Series(completeness_check)
q6 = time_check(new_df)
print(q6)