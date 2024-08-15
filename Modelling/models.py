import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def lasso_regression(df_train, target, random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Performing Lasso regression...")
    
    # Lasso regression with cross-validation for hyperparameter tuning of regularization parameter alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000).fit(X_train_scaled, y_train)
    
    selected_features_lasso = np.where(lasso.coef_ != 0)[0]
    selected_feature_names_lasso = X_train.columns[selected_features_lasso]

    print(f"Selected features by Lasso ({len(selected_feature_names_lasso)}): {selected_feature_names_lasso}")
    
    # collect variable importance
    variable_importance_dict = {
        "Feature": selected_feature_names_lasso,
        "Coefficient": lasso.coef_[selected_features_lasso]
    }
    variable_importance_df = pd.DataFrame(variable_importance_dict)
    variable_importance_df = variable_importance_df.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\nVariable Importance (Lasso):")
    print(variable_importance_df)
    
    return lasso, variable_importance_df

def lasso_feature_selection_linear_regression(df_train, target, random_state=42):
    # Split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # Standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Performing Lasso regression for feature selection...")
    
    # Lasso regression with cross-validation for hyperparameter tuning of regularization parameter alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000).fit(X_train_scaled, y_train)
    
    selected_features_lasso = np.where(lasso.coef_ != 0)[0]
    selected_feature_names_lasso = X_train.columns[selected_features_lasso]

    print(f"Selected features by Lasso ({len(selected_feature_names_lasso)}): {selected_feature_names_lasso}")
    
    # If no features are selected by Lasso, raise an exception
    if len(selected_feature_names_lasso) == 0:
        raise ValueError("No features were selected by Lasso. Try adjusting the Lasso parameters.")
    
    # Collect variable importance
    variable_importance_dict = {
        "Feature": selected_feature_names_lasso,
        "Coefficient": lasso.coef_[selected_features_lasso]
    }
    variable_importance_df = pd.DataFrame(variable_importance_dict)
    variable_importance_df = variable_importance_df.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\nVariable Importance (Lasso):")
    print(variable_importance_df)
    
    # Use only the selected features for the linear regression model
    X_train_selected = X_train[selected_feature_names_lasso]

    # Fit linear regression model with selected features
    linear_regression_model = LinearRegression().fit(X_train_selected, y_train)
    
    print("\nFitted Linear Regression model with selected features.")
    
    return linear_regression_model, variable_importance_df, selected_feature_names_lasso


def random_forest_regression(df_train, target, random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # define the model
    rf = RandomForestRegressor(random_state=random_state)
    
    # define a pipeline that includes 20 most important feature selection and model training
    pipeline = Pipeline([
    ('feature_selection', SelectFromModel(rf, max_features=20)),
    ('rf', rf)
    ])
    
    # define the hyperparameter grid
    param_grid = {
        'rf__n_estimators': [ 200, 500],
        'rf__max_depth': [ 20, 30],
        'rf__min_samples_split': [5, 10],
        'rf__min_samples_leaf': [2, 4]
        }
    
    # perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # best model
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # feature importance
    selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    feature_importance_dict = {
        "Feature": selected_feature_names,
        "Importance": best_model.named_steps['rf'].feature_importances_
    }
    feature_importance_df = pd.DataFrame(feature_importance_dict)
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\nFeature Importance (Random Forest):")
    print(feature_importance_df)
    
    return best_model, feature_importance_df

def decision_tree_regression(df_train, target, random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # define the model
    dt = DecisionTreeRegressor(random_state=random_state)
    
    # define a pipeline that includes feature selection and model training
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(dt, max_features=20)),
        ('dt', dt)
    ])
    
    # define the hyperparameter grid
    param_grid = {
        'dt__max_depth': [10, 20, 30],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4]
    }
    
    # perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # best model
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # feature importance
    selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    feature_importance_dict = {
        "Feature": selected_feature_names,
        "Importance": best_model.named_steps['dt'].feature_importances_
    }
    feature_importance_df = pd.DataFrame(feature_importance_dict)
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\nFeature Importance (Decision Tree):")
    print(feature_importance_df)
    
    return best_model, feature_importance_df


def evaluation_metrics(model_name, model, df_train, df_test, target, random_state=42, scaling = False, outcome_transformation = "None"):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]
    
    # in case of a model with standardization, standardize train and test set again
    if scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    print("Predicting on the training and test data...")

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # in case of outcome transformation, revert to have original scale of data
    if outcome_transformation == "log":
        y_train_pred = np.exp(y_train_pred)
        y_test_pred = np.exp(y_test_pred)
        y_train = np.exp(y_train)
        y_test = np.exp(y_test)
        
    print("Evaluating the model...")
    
    # evaluating on train data
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    # evaluating on test data
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    # collect metrics
    results_dict = {
    "Model": [model_name],
    "Training RMSE": [f"{train_rmse:.2f}"],
    "Training MAE": [f"{train_mae:.2f}"],
    "Training MAPE": [f"{train_mape:.2f}"],
    "Testing RMSE": [f"{test_rmse:.2f}"],
    "Testing MAE": [f"{test_mae:.2f}"],
    "Testing MAPE": [f"{test_mape:.2f}"]
    }

    results_df = pd.DataFrame(results_dict)

    return results_df