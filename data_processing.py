from sklearn.model_selection import train_test_split as tts

def find_constant_columns(df):
    """
    This function takes a dataframe and returns the colums that contain a single value

    Inputs:
        df (pandas dataframe): the dataframe to be used

    Returns:
        list (list): list of columns containing a single value
    """

    constant_cols = []
    for column in df.columns:
        unique_values = df[column].unique()
        if len(unique_values) ==1:
            constant_cols.append(column)
    
    return constant_cols

def delete_constant_columns(df, cols):
    """
    This funciton takes a dataframe and returns the dataframe with the specified columns dropped

    Inputs:
        df (pandas dataframe): the dataframe to be used
        cols (list): a list of the columns to be dropped

    Returns:
        df (pandas dataframe): dataframe with the columns dropped
    
    
    
    """
    df = df.drop(cols, axis=1)
    return df

def cols_with_few_values(df, threshold):
    """
    Function evalues the columns in the provided dataframe to check if there
    if the number of unique values is above the input threshold.

    Inputs:
        df (pandas dataframe): dataframe to be analyzed
        threshold (integer value): minimum number of unique values needed to not flag the column
    Returns:
        list (list): list of columns with unique values below threshold
    
    """

    few_cols = []
    for column in df.columns:
        unique_values_count = len(df[column].unique())
        if unique_values_count < threshold:
            few_cols.append(column)
    return few_cols

def find_duplicate_rows(df):
    """
    Identifies and returns any duplicate data rows.

    Inputs:
        df (pandas df): dataframe to be analyzed
    Returns:
        df (pandas df): new dataframe containing duplicate rows
    """

    duplicate_rows = df[df.duplicated()]
    return duplicate_rows

def delete_duplicate_rows(df):
    """
    Takes a df and deletes all but the first occurrence of any duplicate rows

    Inputs:
        df (pandas df): dataframe to be analyzed

    Returns:
        df (pandas df): the input df with duplicate rows removed
    """

    df = df.drop_duplicates(keep="first")
    return df

def drop_and_fill(df):
    """
    Drops columns with more than 50% missing values and fills in missing
    values with mean of column

    Inputs:
        df (pandas df): pandas df to analyze
    Returns:
        df (pandas df): input dataframe with above manipulations performed

    """

    cols_to_drop = df.columns[df.isnull().mean() > 0.5]
    df = df.drop(cols_to_drop, axis=1)

    df = df.fillna(df.mean())

    return df

def split_data(df, target_column):
    """
    This function takes a dataframe and a target column and splits the dataframe into a feature and a target dataframe.

    Inputs:
    df (DataFrame): The dataframe to be analyzed
    target_column (str): the target column

    Returns:
    df (dataframe): The dataframe containing the features
    df (dataframe): The dataframe containing the target column
    """
    # Split the dataframe into a feature dataframe and a target dataframe
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
    return (X_train, X_test, y_train, y_test)
