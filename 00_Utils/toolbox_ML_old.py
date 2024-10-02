import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, f_oneway, mannwhitneyu, chi2_contingency
from sklearn.feature_selection import f_regression


def get_features_num_regression_mine(df, target_col, corr_threshold, *, pvalue = None, card = 20, return_values = False):
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
    corr_threshold : float
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
        A list of column names whose correlation with 'target_col' exceeds the 'corr_threshold' threshold.
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
    
    # Validate corr_threshold is a float between 0 and 1
    if not isinstance(corr_threshold, (int, float)) or not (0 <= corr_threshold <= 1):
        print('The "corr_threshold" value must be a number between 0 and 1.')
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
        if abs(corr) > corr_threshold:
            if pvalue is None or p_val <= pvalue:
                features_num.append(col)
                if return_values:
                    all_values[col] = {'corr': corr, 'p_value': p_val}
    

    # Return features_num and, if requested, a DataFrame with correlations and p-values
    if return_values:
        return features_num, pd.DataFrame(all_values).T.sort_values('corr', ascending = False)
    else:
        return features_num
    


def plot_features_num_regression(df, target_col = '', columns = [], corr_threshold = 0, pvalue = None):
    """
    Generates pair plots for selected numeric columns in a DataFrame based on their correlation with a specified target column.
    The columns are filtered by a correlation threshold and optionally a p-value significance level. If the columns list is 
    empty, the numeric columns in the DataFrame are considered. If more than 5 columns are to be plotted, the function splits 
    them into multiple pair plots, including the target column in each plot.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data.
    target_col: str 
        The target column to correlate with other numeric columns. It must be a numeric variable.
    columns: list 
        List of column names to consider for the pair plots. If empty, numeric columns will be automatically selected.
    corr_threshold: float 
        Correlation threshold (default is 0). Only columns with absolute correlation higher than this value will be considered.
    pvalue: float, optional
        Significance level for the correlation test. Only columns with p-value less than this will be considered. Default is None (no p-value check).

    Returns
    -------
    list: 
        List of columns that meet the correlation and p-value conditions.
    """
    
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('The "df" parameter must be a pandas DataFrame.')

    # Validate target column
    if target_col not in df.columns:
        raise ValueError(f'The target column "{target_col}" is not present in the DataFrame.')
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f'The target column "{target_col}" must be numeric.')

    # Validate correlation threshold
    if not isinstance(corr_threshold, (int, float)) or not (0 <= corr_threshold <= 1):
        raise ValueError('The "corr_threshold" value must be a number between 0 and 1.')

    # Validate p-value threshold if provided
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            raise ValueError('The "pvalue" must be None or a number between 0 and 1.')


    # If no columns are provided, automatically select numeric columns from the DataFrame
    if not columns:
        columns = get_features_num_regression(df = df, target_col = target_col, corr_threshold = corr_threshold, pvalue = pvalue)

    # Filter columns based on correlation and p-value (if provided)
    valid_columns = []
    for col in columns:
        if col == target_col:
            continue  # Skip the target column itself

        # Calculate Pearson correlation and p-value between the column and the target column
        corr, p_val = pearsonr(df[col], df[target_col])

        # Check if the correlation meets the threshold
        if abs(corr) > corr_threshold:
            # Check p-value significance if pvalue is provided
            if pvalue is None or p_val <= pvalue:
                valid_columns.append(col)
        # Check if the correlation meets the threshold
        if abs(corr) > corr_threshold:
            # Check p-value significance if pvalue is provided
            if pvalue is None or p_val <= pvalue:
                valid_columns.append(col)
        else:
            # Warn that column does not meet the required correlation threshold
            print(f'"{col}" did not meet the correlation threshold of {corr_threshold}.')
            # Ask if you want to remove the column or continue anyway
            question = input(f'Do you want to remove "{col}" from the columns list or continue anyway? Type "remove" or "continue"').strip().lower()
            if question == 'continue':
                valid_columns.append(col) # adds column to valid_cols list if user types continue
            elif question == 'remove':
                print(f'"{col}" was removed from columns list')
                continue

    # If no valid columns remain after filtering, return an empty list
    if not valid_columns:
        print('No columns meet the correlation and p-value criteria.')
        return []

    # Ensure the target column is not included in the pairplot columns
    valid_columns = [col for col in valid_columns if col != target_col]

    # Plot the pair plots in groups of 5 columns (including target_col)
    for i in range(0, len(valid_columns), 4):
        cols_to_plot = [target_col] + valid_columns[i:i + 4]
        sns.pairplot(df, vars = cols_to_plot, hue = target_col)
        plt.show()

    # Return the list of valid columns
    return valid_columns

def get_features_cat_regression(df, target_col, pvalue = 0.05, card = 20):
    """
    Identifies categorical columns in a DataFrame that have a statistically significant relationship with a specified numeric target column.
    The function automatically chooses the appropriate test: ANOVA for categorical columns with more than two categories, and Mann-Whitney U for binary categorical columns.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data.
    target_col: str
        The numeric target column used to test the relationship with categorical columns. This must be a numeric continuous variable with high cardinality.
    pvalue: float, optional 
        The significance level (default is 0.05) for statistical tests. Columns with p-values less than this will be considered significant.
    card: int, optional 
        The maximum percentage of unique values a column can have to be considered categorical (default is 20).

    Returns
    -------
    significant_categorical_features: list
        A list of categorical columns that have a statistically significant relationship with the target column, based on the specified p-value.
    """

    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print('The first argument must be a Pandas DataFrame.')
        return None

    # Validate the target column exists in the DataFrame
    if target_col not in df.columns:
        print(f"The target column '{target_col}' must be present in the DataFrame.")
        return None

    # Validate target_col and card are numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f'The column "{target_col}" must be numeric.')
        return None
    
    # Validate the target column has high cardinality
    percentage_card = df[target_col].nunique() * 100
    if percentage_card <= card:
        print(f'The column "{target_col}" does not have sufficient cardinality. More than {card}% of unique values are required.')
        return None

    # Validate the pvalue parameter
    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print('"pvalue" must be a number between 0 and 1.')
        return None

    # Initialize a list to store categorical features that have a significant relationship with the target column
    significant_categorical_features = []
    
    # Initialize list with categorical features from the dataframe
    cat_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
    
    # Validate if there are categorical columns
    if not cat_columns:
        print('No categorical columns found in dataframe.')
        return None

    # Iterate through the columns of the DataFrame
    for col in cat_columns:
        unique_values = df[col].unique()

        # If the column is binary, use Mann-Whitney U test
        if len(unique_values) == 2:
            groupA = df[df[col] == unique_values[0]][target_col]
            groupB = df[df[col] == unique_values[1]][target_col]

            # Perform the Mann-Whitney U test
            p_val = mannwhitneyu(groupA, groupB).pvalue

        else:
            # For columns with more than 2 unique values, use ANOVA (F-test)
            target_by_groups = [df[df[col] == group][target_col] for group in unique_values]

            # Perform the ANOVA test
            p_val = f_oneway(*target_by_groups).pvalue

        # Check if the p-value is below the specified significance threshold
        if p_val <= pvalue:
            significant_categorical_features.append(col)

    # Return the list of significant categorical features
    return significant_categorical_features



def get_features_cat_regression_mine(df, target_col, pvalue = 0.05, card = 20):
    """
    Identifies categorical columns in a DataFrame that have a statistically significant relationship with a specified numeric target column.
    The function automatically chooses the appropriate test: ANOVA for categorical columns with more than two categories, and Mann-Whitney U for binary categorical columns.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data.
    target_col: str
        The numeric target column used to test the relationship with categorical columns. This must be a numeric continuous variable with high cardinality.
    pvalue: float, optional 
        The significance level (default is 0.05) for statistical tests. Columns with p-values less than this will be considered significant.
    card: int, optional 
        The maximum percentage of unique values a column can have to be considered categorical (default is 20).

    Returns
    -------
    significant_categorical_features: list
        A list of categorical columns that have a statistically significant relationship with the target column, based on the specified p-value.
    """

    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print('The first argument must be a Pandas DataFrame.')
        return None

    # Validate the target column exists in the DataFrame
    if target_col not in df.columns:
        print(f"The target column '{target_col}' must be present in the DataFrame.")
        return None

    # Validate target_col is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f'The column "{target_col}" must be numeric.')
        return None

    # Validate the pvalue parameter
    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print('"pvalue" must be a number between 0 and 1.')
        return None

    
    # Initialize list with categorical features from the dataframe
    cat_columns = df.select_dtypes(include = ['object', 'category', 'bool']).columns.tolist()
    
    # Validate if there are categorical columns
    if not cat_columns:
        print('No categorical columns found in dataframe.')
        return None
    
    # Initialize a list to store categorical features that have a significant relationship with the target column
    significant_categorical_features = []

    # Iterate through the columns of the DataFrame
    for cat_col in cat_columns:
        unique_values = df[cat_col].unique()
        percent_card = round(df[cat_col].nunique() / len(df) * 100, 2)

        # If the column is binary, use Mann-Whitney U test
        if len(unique_values) == 2:
            groupA = df[df[cat_col] == unique_values[0]][target_col]
            groupB = df[df[cat_col] == unique_values[1]][target_col]

            # Perform the Mann-Whitney U test
            p_val = mannwhitneyu(groupA, groupB).pvalue
            
        elif len(unique_values) > 2 and percent_card < 5:
            contingency_table = pd.crosstab(df[cat_col], df[target_col])
            _, p, _, _ = chi2_contingency(contingency_table)
            
        else:
            # For columns with high cardinality, use ANOVA (F-test)
            target_by_groups = [df[df[cat_col] == group][target_col] for group in unique_values]

            p_val = f_oneway(*target_by_groups).pvalue

        # Check if the p-value is below the specified significance threshold
        if p_val <= pvalue:
            significant_categorical_features.append(cat_col)

    # Return the list of significant categorical features
    return significant_categorical_features

#TERCERA
def get_features_num_regression(df, target_col, corr_threshold=0.7, pvalue=None):
    """
    Devuelve dos listas con las columnas numéricas del dataframe que están directa e indirectamente
    correlacionadas con la columna designada por "target_col" según el umbral de correlación especificado.
    Además, se pueden filtrar las columnas por su p-value si se especifica.

    Args:
        df: DataFrame de análisis.
        target_col: Nombre de la columna objetivo.
        corr_threshold: Umbral para considerar una correlación significativa.
        pvalue: Umbral para considerar la significancia estadística de la correlación.

    Returns:
        Dos listas: una con columnas directamente correlacionadas y otra con indirectamente correlacionadas.
    """
    # Asegurarse de que target_col es numérica y está presente en el DataFrame
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna objetivo '{target_col}' debe ser numérica y estar presente en el DataFrame.")

    # Filtrar columnas numéricas, excluyendo la target
    columnas_num = [col for col in df.columns if col != target_col and pd.api.types.is_numeric_dtype(df[col])]

    # Preparar DataFrame solo con columnas numéricas (incluyendo target_col para cálculo de correlación)
    df_numericas = df[columnas_num + [target_col]]

    # Calcular correlaciones
    correlaciones = df_numericas.corr()[target_col].drop(target_col)
    
    # Opcional: Calcular p-values si se requiere filtrar por significancia
    if pvalue is not None:
        _, p_values = f_regression(df_numericas[columnas_num], df_numericas[target_col])
        # Filtrar columnas por p-value
        columnas_significativas = correlaciones.index[p_values < pvalue].tolist()
        correlaciones = correlaciones[columnas_significativas]

    # Filtrar por umbral de correlación
    directamente_correlacionadas = correlaciones[correlaciones >= corr_threshold].index.tolist()
    indirectamente_correlacionadas = correlaciones[correlaciones <= -corr_threshold].index.tolist()

    print(f"Columnas correlacionadas positivamente: {directamente_correlacionadas}, con correlación de Pearson >= {corr_threshold} y significancia en coeficientes {100*(1-pvalue)}%")
    print(f"Columnas correlacionadas negativamente: {indirectamente_correlacionadas}, con correlación de Pearson <= {corr_threshold} y significancia en coeficientes {100*(1-pvalue)}%")

    return directamente_correlacionadas, indirectamente_correlacionadas


#CUARTA 
def plot_features_num_regression(df, target_col="", columns=[], corr_threshold=0, pvalue=None):
    """
    Crea un conjunto de pair plots para visualizar las correlaciones entre las columnas numéricas del DataFrame.

    Args:
        df: El DataFrame del que se quiere visualizar las correlaciones.
        target_col: El nombre de la columna objetivo.
        corr_threshold= numbral establecido de correlacion con la target
        pvalue: El valor de p-valor.

    Returns:
        None
    """

    columnas_para_pintar = []
    columnas_umbral_mayor = []
    
    if columns == []:
        columns = df.columns

    #iteramos por la columnas
    for col in columns:
        #si en la iteracion de las columnas del DF y siempre que...
        # se comprube si son numéricas(true) o no son numéricas(false)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # usando el indice de correlación de Pearson y el p-valor(funcion pearsonr)
            # calculamos dichos parametros para target y resto de columnas
            corr, pv = pearsonr(df[col], df[target_col])
            if abs(corr) > corr_threshold:
                columnas_umbral_mayor.append(col)
                if pvalue is None or pv < pvalue:
                    columnas_para_pintar.append(col)

    # Número máximo de gráficas por grupo
    max_graficas_por_grupo = 5

    # Dividir en grupos según el número máximo de gráficas
    len(columnas_para_pintar) // max_graficas_por_grupo
    # En un alista de comprension, iteramos en rango desde 0 hasta el numero de columnas a pintar, por cada grupo maximo establecido
    # creando graficas con columnas maxi de i+ grupo max establecido ( ejem: '0 hasta 0+6)
    columnas = [columnas_para_pintar[i:i+max_graficas_por_grupo] for i in range(0, len(columnas_para_pintar), max_graficas_por_grupo)]

    # iteramos por i y por valor 'corr_threshold' establecido a cada grupo en cada iteración,  creeando pair plots para cada grupo,
    for i, grupo in enumerate(columnas):
        sns.pairplot(data = df, kind = 'scatter', vars=grupo, hue=target_col)
        plt.suptitle(f"Group {i}", y=1.02)# creo nombres de grupo un poco por encima de y, para que no se superponga con la gráfica
        plt.show()
    
    return f"Las columnas con una correlación de Pearson fuerte y con significancia al {100*(1-pvalue)} en la correlación son", columnas_umbral_mayor


#QUINTA 
def get_features_cat_regression(dataframe: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Esta función recibe un dataframe y dos argumentos adicionales: 'target_col' y 'pvalue'.
    
    Parámetros:
    - dataframe: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    
    Devuelve:
    - Una lista con las columnas categóricas cuya relación con 'target_col' es estadísticamente significativa.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el dataframe
    if target_col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica válida en el dataframe.")
        return None
    
    # Comprueba si 'pvalue' es un float válido
    if not isinstance(pvalue, float):
        print("Error: 'pvalue' debería ser un float.")
        return None
    
    # Identifica las columnas categóricas
    cat_columns = dataframe.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Comprueba si hay columnas categóricas
    if not cat_columns:
        print("Error: No se encontraron columnas categóricas en el dataframe.")
        return None
    
    # Realiza pruebas estadísticas y filtra columnas basadas en el valor de p
    selected_columns = []
    for cat_col in cat_columns:
        
        if round(dataframe[cat_col].nunique() / len(dataframe) * 100, 2) < 5: # Menos de 5% de cardinalidad considero target numérico discreto
        
            contingency_table = pd.crosstab(dataframe[cat_col], dataframe[target_col])
            _, p, _, _ = chi2_contingency(contingency_table)
        
        else: # El target es numérico continuo
            _, p = f_oneway(*[dataframe[target_col][dataframe[cat_col] == category] for category in dataframe[cat_col].unique()])
        
        if p < pvalue:
            selected_columns.append(cat_col)
    
    return selected_columns


#SEXTA
def plot_features_cat_regression(dataframe: pd.DataFrame, target_col: str = "", 
                                  columns: list = [], pvalue: float = 0.05, 
                                  with_individual_plot: bool = False) -> list:
    """
    Esta función recibe un dataframe y varios argumentos opcionales para visualizar y analizar la relación
    entre variables categóricas y una columna objetivo en un modelo de regresión.

    Parámetros:
    - dataframe: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - columns: Lista de nombres de columnas categóricas a considerar (por defecto, todas las numéricas).
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    - with_individual_plot: Booleano que indica si se deben incluir gráficos individuales para cada columna (por defecto es False).

    Devuelve:
    - Una lista con las columnas seleccionadas que cumplen con las condiciones de significancia.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el dataframe
    if target_col and (target_col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[target_col])):
        print(f"Error: '{target_col}' no es una columna numérica válida en el dataframe.")
        return None
    
    # Comprueba si 'pvalue' es un float válido
    if not isinstance(pvalue, float):
        print("Error: 'pvalue' debería ser un float.")
        return None
    
    # Comprueba si 'columns' es una lista válida de strings
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        print("Error: 'columns' debería ser una lista de strings.")
        return None
    
    # Comprueba si 'with_individual_plot' es un booleano válido
    if not isinstance(with_individual_plot, bool):
        print("Error: 'with_individual_plot' debería ser un booleano.")
        return None
    
    # Si 'columns' está vacío, utiliza todas las columnas numéricas en el dataframe
    if not columns:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    
    # Filtra columnas basadas en pruebas estadísticas
    selected_columns = get_features_cat_regression(dataframe, target_col, pvalue)
    #selected_columns = list(set(selected_columns) & set(columns))
    
    if not selected_columns:
        print("Ninguna columna cumple con las condiciones especificadas para trazar.")