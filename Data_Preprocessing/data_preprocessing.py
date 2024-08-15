import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import skew

def load_data(path, sheet_name):
    """Load the Excel file and return a DataFrame."""
    return pd.read_excel(path, sheet_name=sheet_name)

def prepare_base_data(df, random_state = 42):
    
    # drop columns with more than 30% missing values and irrelevant columns
    columns_to_drop = ['yRelativeLowerPower.scaled.corr.N4', 'yRelativeLowerPower.N4', 'yRelativeLowerPower.scaled.N4', 
                       'cTOTEXs', 'cTOTEXs_RP3', 'cTOTEXn_RP3', 'NameOrg', 'NameShort', 'dmuName', 'dmu', 'dDateData', 'BNR', 'BNR_NNR']
    df = df.drop(columns=columns_to_drop)
    
    # drop columns with more than 90% zero values
    threshold = 0.9 * len(df)
    sparse_columns_to_drop = [col for col in df.columns if (df[col] == 0).sum() > threshold]
    df = df.drop(columns=sparse_columns_to_drop)
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=random_state)
    
    # scale and impute Data
    scaler = StandardScaler()

    # fit the scaler on the training data and transform both training and test data
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    # initialize the KNN imputer
    imputer = KNNImputer(n_neighbors=3)

    # fit the imputer on the training data and transform both training and test data
    df_train_scaled = imputer.fit_transform(df_train_scaled)
    df_test_scaled = imputer.transform(df_test_scaled)

    # inverse transform to convert the data back to the original scale
    df_train = scaler.inverse_transform(df_train_scaled)
    df_test = scaler.inverse_transform(df_test_scaled)

    # convert the results back to DataFrames
    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_test = pd.DataFrame(df_test, columns=df.columns)    
    
    return df_train, df_test

def apply_transformation(df_train, df_test, feature, target, degree=2, skewness_threshold=0.5, improvement_threshold=0.01):
    """Apply log or polynomial transformation if necessary."""
    # calculate the skewness of the feature
    skewness = skew(df_train[feature].dropna())
    
    # apply log transformation if skewness is above the threshold
    if skewness > skewness_threshold:
        df_train[feature] = np.log1p(df_train[feature])
        df_test[feature] = np.log1p(df_test[feature])
        df_train.rename(columns={feature: feature + "_log"}, inplace=True)
        df_test.rename(columns={feature: feature + "_log"}, inplace=True)
        return
    
    # prepare data for polynomial transformation
    X_train = df_train[[feature]]
    y_train = df_train[target]
    
    # fit a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_train)
    
    # fit a polynomial regression model
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_train)
    
    # compare the R-squared values of the linear and polynomial models
    r2_lin = r2_score(y_train, y_pred_lin)
    r2_poly = r2_score(y_train, y_pred_poly)
    
    # apply polynomial transformation if it significantly improves the R-squared value
    if r2_poly - r2_lin > improvement_threshold:
        df_train[feature] = y_pred_poly
        X_val = df_test[[feature]]
        X_poly_val = poly.transform(X_val)
        df_test[feature] = poly_reg.predict(X_poly_val)
        df_train.rename(columns={feature: feature + "_poly"}, inplace=True)
        df_test.rename(columns={feature: feature + "_poly"}, inplace=True)

def transform_features(df_train, df_test, target, degree=2, skewness_threshold=0.5, improvement_threshold=0.01):
    """Iterate over df_train and df_test to apply transformations."""
    # iterate over all columns except the target
    for feature in df_train.columns:
        if feature != target:
            apply_transformation(df_train, df_test, feature, target, degree, skewness_threshold, improvement_threshold)

    return df_train, df_test

def aggregate_and_sum_by_group(df):
    # fnd unique variable prefixes (variable names before ".N")
    variable_groups = set(col.split('.N')[0] for col in df.columns if '.N' in col)
    
    # ditionary to collect new columns
    new_columns = {}

    for var in variable_groups:
        # identify columns that belong to this variable group for N1-N4 and N5-N7
        n1_n4_cols = [f"{var}.N{i}" for i in range(1, 5) if f"{var}.N{i}" in df.columns]
        n5_n7_cols = [f"{var}.N{i}" for i in range(5, 8) if f"{var}.N{i}" in df.columns]
        
        # sum the columns within each group
        if n1_n4_cols:
            new_columns[f'{var}_agg_N1to4'] = df[n1_n4_cols].sum(axis=1)
        if n5_n7_cols:
            new_columns[f'{var}_agg_N5to7'] = df[n5_n7_cols].sum(axis=1)
        
        # drop the original N1-N7 columns
        df.drop(columns=n1_n4_cols + n5_n7_cols, inplace=True)
    
    # concatenate all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    # filter out possibly correlated existing aggregate columns
    df = df.filter(regex='^(?!.*(tot|sum|^cTOTEXn$)).*').copy()

    return df

def create_variations(df_train, df_test, random_state = 42):
    
    # log transformed features
    df_train_xlog = df_train.copy()
    df_test_xlog = df_test.copy()
    transform_features(df_train_xlog, df_test_xlog, target='cTOTEXn')
    
    # log transformed features and outcome
    df_train_xlog_ylog = df_train_xlog.copy()
    df_test_xlog_ylog = df_test_xlog.copy()
    df_train_xlog_ylog['cTOTEXn_log'] = np.log1p(df_train_xlog_ylog['cTOTEXn'])
    df_test_xlog_ylog['cTOTEXn_log'] = np.log1p(df_test_xlog_ylog['cTOTEXn'])
    df_train_xlog_ylog.drop(columns=['cTOTEXn'], inplace=True)
    df_test_xlog_ylog.drop(columns=['cTOTEXn'], inplace=True)
    
    # log transformed outcome
    df_train_ylog = df_train.copy()
    df_test_ylog = df_test.copy()
    df_train_ylog['cTOTEXn_log'] = np.log1p(df_train_ylog['cTOTEXn'])
    df_test_ylog['cTOTEXn_log'] = np.log1p(df_test_ylog['cTOTEXn'])
    df_train_ylog.drop(columns=['cTOTEXn'], inplace=True)
    df_test_ylog.drop(columns=['cTOTEXn'], inplace=True)
    
    # only aggregated features
    df_train_agg = df_train.copy().filter(regex='(tot|sum)$|^cTOTEXn$')
    df_test_agg = df_test.copy().filter(regex='(tot|sum)$|^cTOTEXn$')
    
    # only aggregated log features
    df_train_agg_log = df_train_xlog_ylog.copy().filter(regex='(tot|sum)|^cTOTEXn_log$')
    df_test_agg_log = df_test_xlog_ylog.copy().filter(regex='(tot|sum)|^cTOTEXn_log$')
    
    # only disaggregated features
    df_train_non_agg = df_train.copy().filter(regex='^(?!.*(tot|sum|^cTOTEXn$)).*')
    df_test_non_agg = df_test.copy().filter(regex='^(?!.*(tot|sum|^cTOTEXn$)).*')
    
    # aggregation only on N1-4 and N5-7
    df_train_group_agg = df_train.copy()
    df_test_group_agg = df_test.copy()
    df_train_group_agg = aggregate_and_sum_by_group(df_train_group_agg)
    df_test_group_agg = aggregate_and_sum_by_group(df_test_group_agg)
    
    df_train_list = [df_train, df_train_xlog, df_train_xlog_ylog, df_train_ylog, df_train_agg, df_train_agg_log, df_train_non_agg, df_train_group_agg]
    df_test_list = [df_test, df_test_xlog, df_test_xlog_ylog, df_test_ylog, df_test_agg, df_test_agg_log, df_test_non_agg, df_test_group_agg]
    
    return df_train_list, df_test_list