import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, f_oneway, mannwhitneyu

def describe_df_extra(df, count = False, na_count = False, sort_na = False): 
    """
    Generates a summary DataFrame describing the input DataFrame's data types, percentage of missing values, number of unique values, cardinality (percentage of unique values), and optionally, the count of non-null values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be described.
    
    count : bool, optional
        If True, includes the count of non-null values in each column (default is False).
    
    Returns
    -------
    df_summary: pd.DataFrame
        A DataFrame with a summary of data types, missing values, unique values, cardinality, and optionally, the count of non-null values for each column.
    
    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    
    ValueError
        If the DataFrame is empty.
    """

    # Validate input type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Input must be a pandas DataFrame, but received {type(df).__name__}.')
    
    # Calculate the length of the DataFrame once
    num_rows = len(df)
    
    # Validate DataFrame length to prevent dividing by 0 later on
    if num_rows == 0:
        raise ValueError('The DataFrame is empty.')
    
    # Calculate data types, missing values percentage, unique values and cardinality
    data_type = df.dtypes
    missings = round(df.isna().sum() / num_rows * 100, 2)
    unique_values = df.nunique()
    cardin = round(unique_values / num_rows * 100, 2)
    
    # Construct the summary DataFrame
    df_summary = pd.DataFrame({
        'DATA_TYPE': data_type,
        '%_MISSINGS': missings,
        'UNIQUE_VALUES': unique_values,
        '%_CARDIN': cardin
    })
    
    # Optionally add the count of non-null values and rearrange the columns
    if count and na_count:
        not_null_count = df.notna().sum()
        na = df.isna().sum()
        df_summary.insert(1, 'NOT-NULL', not_null_count)
        df_summary.insert(2, 'MISSINGS', na)
    elif count:
        not_null_count = df.notna().sum()
        df_summary.insert(1, 'NOT-NULL', not_null_count)
    elif na_count:
        na = df.isna().sum()
        df_summary.insert(1, 'MISSINGS', na)
        
    if sort_na:
        return df_summary.sort_values('%_MISSINGS', ascending = False)
    
    return df_summary

def tipifica_variables_extra(df, umbral_categoria, umbral_continua, *, unique_values = False, cardinality = False):
    """
    Classifies the columns of a DataFrame based on their cardinality and percentage cardinality.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame whose columns will be classified.
    umbral_categoria : int
        The threshold for categorical variables. Columns with unique values less than or equal to this threshold will be classified as 'Categorica'.
    umbral_continua : float
        The threshold for continuous numerical variables, based on the percentage of unique values in the column. 
        If the percentage of unique values is greater than or equal to this threshold, the column is classified as 'Numerica Continua'.
    unique_values : bool, optional (default=False)
        If True, includes the cardinality (number of unique values) of each column in the output DataFrame.
    cardinality : bool, optional (default=False)
        If True, includes the percentage of unique values (cardinality relative to the total number of rows) of each column in the output DataFrame.

    Returns
    -------
    df_type : pandas.DataFrame
        A DataFrame with columns 'nombre_variable', 'tipo_sugerido', and optionally 'cardinalidad' and '%_cardinalidad'based on the input flags (unique_values and cardinality).
        The DataFrame provides the column names and their suggested type classification.
    
    Raises
    ------
    TypeError
        If the input `df` is not a pandas DataFrame, or if `umbral_categoria` is not an integer, or `umbral_continua` is not a float.
    """
    
    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Parameter df must be a pandas DataFrame, but received {type(df).__name__}.')
    if not isinstance(umbral_categoria, (int, float)):
        raise TypeError(f'Parameter umbral_categoria must be int, but received {type(umbral_categoria).__name__}.')
    if not isinstance(umbral_continua, (int, float)):
        raise TypeError(f'Parameter umbral_continua must be float, but received {type(umbral_continua).__name__}.')
    
    # Change types if needed
    if isinstance(umbral_categoria, float):
        umbral_categoria = int(umbral_categoria)
    if isinstance(umbral_continua, int):
        umbral_categoria = float(umbral_categoria)

    # Get the number of rows in the DataFrame
    num_rows = len(df) 
    
    # Lists to store column names and their suggested type
    col_name = []
    suggested_type = []
    
    # Lists to store cardinality and percentage, if required
    if unique_values:
        unique_list = []
    if cardinality:
        cardinality_list = []

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Calculate unique and percentage cardinality
        unique = df[col].nunique()
        percentage_cardinality = unique / num_rows * 100
        
        # Classify the variable based on cardinality and percentage cardinality
        if unique == 2:
            type_classification = 'Binaria'
        elif unique < umbral_categoria:
            type_classification = 'Categorica'
        else:
            type_classification = 'Numerica Continua' if percentage_cardinality >= umbral_continua else 'Numerica Discreta'
        
        # Add column name and its classification to their respective lists
        col_name.append(col)
        suggested_type.append(type_classification)
        
        # If unique_values is True, store the numeber of unique values
        if unique_values:
            unique_list.append(unique)
        # If cardinality is True, store the percentage cardinality, rounded to 2 decimal places
        if cardinality:
            cardinality_list.append(round(percentage_cardinality, 2))
    
    # Create a DataFrame with column names and their suggested types
    df_type = pd.DataFrame({'nombre_variable': col_name, 'tipo_sugerido': suggested_type})
    
    # Insert additional columns based on the flags: unique_values and cardinality
    if unique_values and cardinality:
        df_type.insert(1, 'n_unique', unique_list)
        df_type.insert(2, '%_cardinalidad', cardinality_list)
    elif unique_values:
        df_type.insert(1, 'n_unique', unique_list)
    elif cardinality:
        df_type.insert(1, '%_cardinalidad', cardinality_list)

    # Return the final DataFrame with the classifications
    return df_type

def get_features_num_regression_extra(df, target_col, umbral_corr, *, pvalue = None, card = 20, return_values = False):
    """
    Identifies numeric columns in a DataFrame whose correlation with 'target_col' exceeds a specified
    correlation threshold (absolute value) and, optionally, passes a statistical significance test based on the p-value.
    Optionally, returns detailed information about the correlations and p-values of the filtered features.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_col : str
        The target column name to calculate correlation with other numeric columns.
    umbral_corr : float
        The correlation threshold to filter columns (absolute value between 0 and 1).
    pvalue : float, optional
        The significance level to filter statistically significant correlations (between 0 and 1). Default is None.
    card : int, float, optional
        The minimum cardinality percentage required for 'target_col' to be considered continuous. Default is 20.
    return_values : bool, optional
        If True, returns a DataFrame with correlations and p-values for each filtered column. Default is False.

    Returns
    -------
    features_num : list
        A list of column names whose correlation with 'target_col' exceeds the 'umbral_corr' threshold.
    all_values : pandas.DataFrame, optional
        If `return_values=True`, returns a DataFrame containing the correlation and p-value for each selected feature, 
        sorted by the correlation in descending order. Columns are named 'corr' and 'p_value'.
    """
    
    # Validate the DataFrame
    if not isinstance(df, pd.DataFrame):
        print('The "df" parameter must be a pandas DataFrame.')
        return None
    
    # Validate target_col exists in the DataFrame
    if target_col not in df.columns:
        print(f'The column "{target_col}" is not present in the DataFrame.')
        return None
    
    # Validate target_col and card are numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f'The column "{target_col}" must be numeric.')
        return None
    
    if not isinstance(card, (int, float)):
        print('The "card" parameter must be a number (int or float).')
        return None
    
    # Validate target_col has high cardinality
    percentage_card = df[target_col].nunique() * 100
    if percentage_card <= card:
        print(f'The column "{target_col}" does not have sufficient cardinality. More than {card}% of unique values are required.')
        return None
    
    # Validate umbral_corr is a float between 0 and 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print('The "umbral_corr" value must be a number between 0 and 1.')
        return None
    
    # Validate pvalue is a float between 0 and 1 if provided
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print('The "pvalue" must be "None" or a number (float) between 0 and 1.')
            return None
    
    # Select numeric columns excluding the target column
    numeric_cols = df.select_dtypes(include = [int, float]).columns.difference([target_col])
    
    # Initialize the list to store selected features
    features_num = []
    
    # Initialize dictionary to store all correlations and p-values if return_values is True
    if return_values:
        all_values = {}
    
    # Calculate correlations and filter by threshold
    for col in numeric_cols:
        corr, p_val = pearsonr(df[col], df[target_col])
        if abs(corr) > umbral_corr:
            if pvalue is None or p_val <= pvalue:
                features_num.append(col)
                if return_values:
                    all_values[col] = {'corr': corr, 'p_value': p_val}
    

    # Return features_num and, if requested, a DataFrame with correlations and p-values
    if return_values:
        return features_num, pd.DataFrame(all_values).T.sort_values('corr', ascending = False)
    else:
        return features_num

