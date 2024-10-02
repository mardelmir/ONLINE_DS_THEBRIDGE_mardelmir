import numpy as np
import pandas as pd

from scipy.stats import pearsonr, f_oneway, mannwhitneyu, chi2_contingency
from sklearn.feature_selection import f_regression

def describe_df(df, count = False, na_count = False, sort_by = None, ascending = True):
    """
    Generates a summary DataFrame describing key statistics of the input DataFrame, including:
    
    - Data types.
    - Percentage of missing values.
    - Number of unique values.
    - Cardinality (percentage of unique values).
    
    Optionally, it can also include the count of non-null values and the absolute number of missing values.
    
    The summary DataFrame can be sorted by a specified column, either by name or index, in ascending or descending order.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be described.
    
    count : bool, optional, default False
        If True, includes the count of non-null values in each column.
        
    na_count : bool, optional, default False
        If True, includes the absolute number of missing values in each column.
    
    sort_by : str or int, optional, default None
        Specifies the column by which the summary DataFrame should be sorted.
        Can be a column name (str) or an index (int). For example, 'Data type' or 0.
        If None, no sorting is applied.
    
    ascending : bool, optional, default True
        Specifies the sorting order. If True, sorts in ascending order. If False, sorts in descending order.

    Returns
    -------
    df_summary : pd.DataFrame
        A DataFrame containing the following columns:
        - 'Data type': The data type of each column.
        - '% Missings': The percentage of missing values.
        - 'Unique values': The count of unique values in each column.
        - '% Cardinality': The cardinality, or percentage of unique values.
        - Optionally, 'Not-Null' and 'Missings', based on the `count` and `na_count` flags.
    
    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame or if `sort_by` is neither a string nor an integer.
    
    ValueError
        If the DataFrame is empty, or if the specified `sort_by` column name or index is invalid.
    """

    # Validate input type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Input must be a pandas DataFrame, but received {type(df).__name__}.')


    # Calculate the length of the DataFrame once for efficiency
    num_rows = len(df)
    
    # Validate DataFrame length to prevent division by 0
    if num_rows == 0:
        raise ValueError('The DataFrame is empty.')
    
    # Calculate core statistics for the DataFrame
    data_type = df.dtypes
    missings = (df.isna().sum() / num_rows * 100).round(2)  # Optimized to avoid rounding twice
    unique_values = df.nunique()
    cardin = (unique_values / num_rows * 100).round(2)
    
    # Create the summary DataFrame
    df_summary = pd.DataFrame({
        '#': df.columns.argsort(),
        'Data type': data_type,
        '% Missings': missings,
        'Unique values': unique_values,
        '% Cardinality': cardin
    })
    
    # Optionally add the count of non-null values and missing values
    if count or na_count:
        not_null_count = df.notna().sum() if count else None
        na = df.isna().sum() if na_count else None
        
        if count:
            df_summary.insert(2, 'Not-Null', not_null_count)
        if na_count:
            df_summary.insert(2, 'Missings', na)
    
    # Sort by column if sort_by is provided
    if sort_by is not None:
        if isinstance(sort_by, str):
            # Check if the provided column name exists in the DataFrame
            if sort_by not in df_summary.columns:
                raise ValueError(f'Column name "{sort_by}" is not valid.')
            df_summary = df_summary.sort_values(by = sort_by, ascending = ascending)
        elif isinstance(sort_by, int):
            # Check if the provided index is within bounds
            if sort_by < 0 or sort_by >= len(df_summary.columns):
                raise ValueError(f'Column index {sort_by} is out of bounds.')
            df_summary = df_summary.sort_values(by = df_summary.columns[sort_by], ascending = ascending)
        else:
            raise TypeError(f'"sort_by" must be a string (column name) or an integer (column index).')
    
    return df_summary

def typify_variables(df, umbral_categoria, umbral_continua, *, unique_values = False, cardinality = False):
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
    
def get_features_num_regression_mine(df, target_col, corr_threshold, *, pvalue = None, card = 20, return_values = False):
    """
    Identifies numeric columns in a DataFrame whose correlation with 'target_col' exceeds a specified
    threshold and optionally passes a significance test based on p-value.
    Returns detailed information about the correlations and p-values if requested.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_col : str
        The column name to calculate correlation with other numeric columns.
    corr_threshold : float
        The correlation threshold (absolute value between 0 and 1) to filter columns.
    pvalue : float, optional
        The significance level to filter statistically significant correlations (between 0 and 1). Default is None.
    card : int or float, optional
        The minimum cardinality percentage required for 'target_col' to be considered continuous. Default is 20.
    return_values : bool, optional
        If True, returns a DataFrame with correlations and p-values for each filtered column. Default is False.

    Returns
    -------
    features_num : list
        A list of column names whose correlation with 'target_col' exceeds the 'corr_threshold'.
    all_values : pandas.DataFrame, optional
        If `return_values=True`, returns a DataFrame containing correlations and p-values for each feature.
    """
    
    # Validate the DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise ValueError('The "df" parameter must be a pandas DataFrame.')

    # Check if the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f'The column "{target_col}" is not present in the DataFrame.')

    # Ensure the target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f'The column "{target_col}" must be numeric.')

    # Ensure the cardinality parameter is valid
    if not isinstance(card, (int, float)):
        raise ValueError('The "card" parameter must be a number (int or float).')

    # Check if target column has sufficient cardinality
    percentage_card = df[target_col].nunique() * 100 / len(df)
    if percentage_card <= card:
        raise ValueError(f'The column "{target_col}" does not have sufficient cardinality (more than {card}% required).')

    # Ensure the correlation threshold is valid
    if not isinstance(corr_threshold, (int, float)) or not (0 <= corr_threshold <= 1):
        raise ValueError('The "corr_threshold" must be a number between 0 and 1.')

    # Validate pvalue if provided
    if pvalue is not None and (not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1)):
        raise ValueError('The "pvalue" must be a number between 0 and 1 or None.')

    # Select numeric columns, excluding the target column
    numeric_cols = df.select_dtypes(include = [np.number]).columns.difference([target_col])

    # Initialize the list to store selected numeric features
    features_num = []

     # Initialize dictionary to store all correlations and p-values if return_values is True
    if return_values:
        all_values = {}

    # Calculate correlations and p-values for filtering
    for col in numeric_cols:
        corr, p_val = pearsonr(df[col], df[target_col])
        if abs(corr) > corr_threshold and (pvalue is None or p_val <= pvalue):
            features_num.append(col)
            if return_values:
                all_values[col] = {'corr': corr, 'p_value': p_val}

    # Return selected features and optionally the correlation/p-value DataFrame
    if return_values:
        return features_num, pd.DataFrame(all_values).T.sort_values('corr', ascending = False)
    else:
        return features_num
   
def get_features_cat_regression_mine(df, target_col, pvalue = 0.05, card = 20, return_values = False):
    """
    Identifies categorical columns in a DataFrame that have a statistically significant relationship with a numeric target column.
    For binary categorical columns, it uses the Mann-Whitney U test, while for others, it applies ANOVA.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame containing the data.
    target_col: str
        The numeric target column used for testing relationships with categorical columns.
    pvalue: float, optional
        The significance level (default is 0.05). Columns with p-values below this threshold are considered significant.
    card: int or float, optional
        The percentage threshold for column uniqueness to consider it categorical (default is 20).
    return_values: bool, optional
        If True, returns a DataFrame with p-values for each significant feature.

    Returns
    -------
    significant_categorical_features : list
        A list of categorical columns that have a significant relationship with the target column.
    all_values : pandas.DataFrame, optional
        If `return_values=True`, returns a DataFrame containing p-values for each feature.
    """
    
    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('The first argument must be a pandas DataFrame.')

    # Ensure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"The target column '{target_col}' must be present in the DataFrame.")

    # Ensure target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f'The column "{target_col}" must be numeric.')

    # Validate pvalue parameter
    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        raise ValueError('"pvalue" must be a number between 0 and 1.')


    # Select categorical columns
    cat_columns = df.select_dtypes(include = ['object', 'category', 'bool']).columns.tolist()

    # Check if there are categorical columns in the DataFrame
    if not cat_columns:
        raise ValueError('No categorical columns found in the DataFrame.')

    # List to store significant categorical features
    significant_categorical_features = []
    
    if return_values:
        all_values = {}

    # Iterate through categorical columns
    for cat_col in cat_columns:
        unique_values = df[cat_col].unique()
        percent_card = df[cat_col].nunique() / len(df) * 100

        # Mann-Whitney U test for binary categorical columns
        if len(unique_values) == 2:
            groupA = df[df[cat_col] == unique_values[0]][target_col]
            groupB = df[df[cat_col] == unique_values[1]][target_col]
            p_val = mannwhitneyu(groupA, groupB).pvalue

        # Chi-squared test for low-cardinality categorical columns
        elif len(unique_values) > 2 and percent_card < 5:
            contingency_table = pd.crosstab(df[cat_col], df[target_col])
            p_val = chi2_contingency(contingency_table).pvalue

        # ANOVA for high-cardinality categorical columns
        else:
            target_by_groups = [df[df[cat_col] == group][target_col] for group in unique_values]
            p_val = f_oneway(*target_by_groups).pvalue

        # Append to results if the p-value is below the threshold
        if p_val <= pvalue:
            significant_categorical_features.append(cat_col)
            if return_values:
                all_values[cat_col] = {'p_value': p_val}

    # Return the list of significant features and optionally the p-value DataFrame
    if return_values:
        return significant_categorical_features, pd.DataFrame(all_values).T.sort_values('p_value', ascending = False)
    else:
        return significant_categorical_features
